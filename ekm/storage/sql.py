from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, or_, text, select
import uuid
import json
import logging
import numpy as np
from .base import BaseStorage
from ..core.models import Workspace, Episode, AKU, AKURelationship, GKU, gku_aku_association

# Import pgvector if available
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False

def _unit_vector(embedding) -> Optional[np.ndarray]:
    """An embedding (list, array, JSON string or pgvector value) scaled to length 1, or None."""
    if embedding is None:
        return None
    if isinstance(embedding, (bytes, bytearray)):
        embedding = embedding.decode('utf-8')
    if isinstance(embedding, str):
        try:
            embedding = json.loads(embedding)
        except json.JSONDecodeError:
            return None
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec)) if vec.size else 0.0
    if norm == 0.0:
        return None
    return vec / norm


class SQLStorage(BaseStorage):
    def __init__(self, db: AsyncSession):
        self.db = db
        # workspace_id -> (episode ids, unit-vector matrix); see _episode_index
        self._episode_indexes: Dict[str, Any] = {}

    def _serialize_embedding(self, embedding):
        if isinstance(embedding, (list, np.ndarray)):
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        elif isinstance(embedding, str):
            return json.loads(embedding)
        return embedding

    def _deserialize_embedding(self, embedding):
        if embedding is None:
            return None
        if isinstance(embedding, str):
            try:
                return json.loads(embedding)
            except json.JSONDecodeError:
                return []
        if isinstance(embedding, (bytes, bytearray)):
             try:
                 return json.loads(embedding.decode('utf-8'))
             except Exception as e:
                 logging.warning(f"Failed to decode embedding bytes: {e}")
                 return []
        if isinstance(embedding, (list, tuple, np.ndarray)):
            return list(embedding)
        return []

    def _standardize_data(self, data):
        """Recursively convert NumPy types to standard Python types for JSON serialization."""
        if isinstance(data, dict):
            return {str(k): self._standardize_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._standardize_data(i) for i in data]
        elif isinstance(data, (np.int64, np.int32, np.integer)):
            return int(data)
        elif isinstance(data, (np.float64, np.float32, np.floating)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        return data

    async def get_workspace(self, workspace_id: str) -> Dict[str, Any]:
        result = await self.db.execute(
            select(Workspace).where(Workspace.id == uuid.UUID(workspace_id))
        )
        workspace = result.scalar_one_or_none()
        if not workspace:
            return None
        return {'id': str(workspace.id), 'name': workspace.name}

    async def save_episode(self, workspace_id: str, content: str, summary: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        episode = Episode(
            workspace_id=uuid.UUID(workspace_id),
            content=content,
            summary=summary,
            embedding=self._serialize_embedding(embedding),
            metadata=metadata
        )
        self.db.add(episode)
        await self.db.commit()
        await self.db.refresh(episode)  # Refresh to get ID if not already set

        if not self._use_pgvector():
            self._add_to_episode_index(workspace_id, str(episode.id), embedding)

        return str(episode.id)

    async def save_akus(self, workspace_id: str, episode_id: Optional[str], akus: List[Dict[str, Any]]) -> List[str]:
        ids = []
        for aku_data in akus:
            aku = AKU(
                workspace_id=uuid.UUID(workspace_id),
                episode_id=uuid.UUID(episode_id) if episode_id else None,
                content=aku_data['content'],
                embedding=self._serialize_embedding(aku_data['embedding']),
                aku_metadata=aku_data.get('metadata', {})
            )
            self.db.add(aku)
            ids.append(aku)
        await self.db.commit()
        return [str(a.id) for a in ids]

    async def save_relationships(self, workspace_id: str, relationships: List[Dict[str, Any]]):
        for rel_data in relationships:
            rel = AKURelationship(
                workspace_id=uuid.UUID(workspace_id),
                source_aku_id=uuid.UUID(rel_data['source_aku_id']),
                target_aku_id=uuid.UUID(rel_data['target_aku_id']),
                semantic_similarity=rel_data.get('semantic_similarity', 0.0),
                temporal_proximity=rel_data.get('temporal_proximity', 0.0),
                causal_weight=rel_data.get('causal_weight', 0.0),
                edge_attributes=rel_data.get('edge_attributes', {})
            )
            self.db.add(rel)
        await self.db.commit()

    async def find_similar_episodes(self, workspace_id: str, embedding: List[float], threshold: float, limit: int) -> List[Dict[str, Any]]:
        """
        Episodes of this workspace whose cosine similarity to `embedding` is at
        least `threshold`, best first. 'similarity' is cosine on every backend
        (pgvector or not), so scores from here compare directly with fact scores.
        """
        query = _unit_vector(embedding)
        if query is None:
            return []

        if self._use_pgvector():
            sql_query = text("""
                SELECT id, content, summary, created_at,
                       (embedding <=> :query_embedding) AS distance
                FROM ekm_episodes
                WHERE workspace_id = :workspace_id
                ORDER BY distance ASC
                LIMIT :limit
            """)
            result = await self.db.execute(sql_query, {
                'query_embedding': json.dumps(query.tolist()),
                'workspace_id': uuid.UUID(workspace_id),
                'limit': limit
            })
            results = []
            for row in result.fetchall():
                sim = 1.0 - float(row[4])  # <=> is cosine distance
                if sim >= threshold:
                    results.append({'id': str(row[0]), 'content': row[1], 'summary': row[2],
                                    'created_at': row[3], 'similarity': sim})
            return results

        ids, matrix = await self._episode_index(workspace_id)
        if not ids or matrix.shape[1] != query.shape[0]:
            return []
        sims = matrix @ query
        order = [int(i) for i in np.argsort(-sims) if sims[i] >= threshold][:limit]
        if not order:
            return []
        rows = await self.db.execute(
            select(Episode).where(
                Episode.id.in_([uuid.UUID(ids[i]) for i in order]),
                Episode.workspace_id == uuid.UUID(workspace_id)
            )
        )
        by_id = {str(ep.id): ep for ep in rows.scalars().all()}
        results = []
        for i in order:
            ep = by_id.get(ids[i])
            if ep is not None:
                results.append({'id': ids[i], 'summary': ep.summary, 'content': ep.content,
                                'created_at': ep.created_at, 'similarity': float(sims[i])})
        return results

    async def _episode_index(self, workspace_id: str):
        """
        (ids, unit-vector matrix) of one workspace's episodes, loaded from the
        database on first use and kept per workspace. The previous single index
        was filled with whichever workspace was queried first and searched
        without a workspace filter, so other workspaces' episodes leaked in.
        """
        cached = self._episode_indexes.get(workspace_id)
        if cached is not None:
            return cached
        result = await self.db.execute(
            select(Episode.id, Episode.embedding).where(Episode.workspace_id == uuid.UUID(workspace_id))
        )
        ids, rows = [], []
        for ep_id, emb in result.all():
            vec = _unit_vector(emb)
            if vec is None or (rows and vec.shape[0] != rows[0].shape[0]):
                continue
            ids.append(str(ep_id))
            rows.append(vec)
        matrix = np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)
        self._episode_indexes[workspace_id] = (ids, matrix)
        return ids, matrix

    def _add_to_episode_index(self, workspace_id: str, episode_id: str, embedding) -> None:
        """Append to a workspace's cached index; if none is cached it is built on next search."""
        cached = self._episode_indexes.get(workspace_id)
        vec = _unit_vector(embedding)
        if cached is None or vec is None:
            return
        ids, matrix = cached
        if matrix.size and matrix.shape[1] != vec.shape[0]:
            return
        matrix = vec.reshape(1, -1) if not matrix.size else np.vstack([matrix, vec])
        self._episode_indexes[workspace_id] = (ids + [episode_id], matrix)

    def _use_pgvector(self) -> bool:
        return PGVECTOR_AVAILABLE and self.db.get_bind().dialect.name == 'postgresql'

    async def find_similar_akus(self, workspace_id: str, embedding: List[float], threshold: float, limit: int) -> List[Dict[str, Any]]:
        """Active facts of this workspace with cosine similarity >= threshold, best first."""
        query = _unit_vector(embedding)
        if query is None:
            return []

        if self._use_pgvector():
            sql_query = text("""
                SELECT id, content, episode_id,
                       (embedding <=> :query_embedding) AS distance
                FROM ekm_akus
                WHERE workspace_id = :workspace_id
                  AND is_archived = FALSE
                ORDER BY distance ASC
                LIMIT :limit
            """)
            result = await self.db.execute(sql_query, {
                'query_embedding': json.dumps(query.tolist()),
                'workspace_id': uuid.UUID(workspace_id),
                'limit': limit
            })
            results = []
            for row in result.fetchall():
                sim = 1.0 - float(row[3])  # <=> is cosine distance
                if sim >= threshold:
                    results.append({'id': str(row[0]), 'content': row[1],
                                    'episode_id': str(row[2]) if row[2] else None,
                                    'similarity': sim})
            return results

        result = await self.db.execute(
            select(AKU).where(
                AKU.workspace_id == uuid.UUID(workspace_id),
                AKU.is_archived == False
            )
        )
        results = []
        for aku in result.scalars().all():
            vec = _unit_vector(aku.embedding)
            if vec is None or vec.shape[0] != query.shape[0]:
                continue
            sim = float(vec @ query)
            if sim >= threshold:
                results.append({
                    'id': str(aku.id),
                    'content': aku.content,
                    'episode_id': str(aku.episode_id) if aku.episode_id else None,
                    'similarity': sim
                })
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    async def get_aku_relationships(self, workspace_id: str, aku_ids: List[str]) -> List[Dict[str, Any]]:
        uuids = [uuid.UUID(i) for i in aku_ids]
        result = await self.db.execute(
            select(AKURelationship).where(
                and_(
                    or_(AKURelationship.source_aku_id.in_(uuids), AKURelationship.target_aku_id.in_(uuids)),
                    AKURelationship.workspace_id == uuid.UUID(workspace_id)
                )
            )
        )
        rels = result.scalars().all()
        return [{
            'source_aku_id': str(r.source_aku_id),
            'target_aku_id': str(r.target_aku_id),
            'semantic_similarity': r.semantic_similarity,
            'temporal_proximity': r.temporal_proximity,
            'causal_weight': r.causal_weight
        } for r in rels]

    async def get_akus_by_ids(self, aku_ids: List[str]) -> List[Dict[str, Any]]:
        uuids = [uuid.UUID(i) for i in aku_ids]
        result = await self.db.execute(
            select(AKU).where(
                AKU.id.in_(uuids),
                AKU.is_archived == False
            )
        )
        akus = result.scalars().all()
        return [{
            'id': str(a.id),
            'content': a.content,
            'embedding': a.embedding,
            'aku_metadata': a.aku_metadata,
            'episode_id': str(a.episode_id) if a.episode_id else None
        } for a in akus]

    async def archive_akus(self, aku_ids: List[str]):
        """Mark AKUs as archived."""
        from sqlalchemy import update
        uuids = [uuid.UUID(i) for i in aku_ids]
        await self.db.execute(
            update(AKU)
            .where(AKU.id.in_(uuids))
            .values(is_archived=True)
        )
        await self.db.commit()

    async def save_gku(self, workspace_id: str, name: str, description: str = "", 
                      centroid_embedding: Optional[List[float]] = None, 
                      pattern_signature: Optional[Dict[str, Any]] = None,
                      cluster_metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a new Global Knowledge Unit."""
        gku = GKU(
            id=uuid.UUID(str(uuid.uuid4())),
            workspace_id=uuid.UUID(workspace_id),
            name=name,
            description=description,
            centroid_embedding=self._serialize_embedding(centroid_embedding) if centroid_embedding is not None else None,
            pattern_signature=self._standardize_data(pattern_signature) if pattern_signature else None,
            cluster_metadata=self._standardize_data(cluster_metadata) if cluster_metadata else {}
        )
        self.db.add(gku)
        await self.db.commit()
        await self.db.refresh(gku)
        return str(gku.id)

    async def get_gku(self, gku_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific GKU by ID."""
        result = await self.db.execute(
            select(GKU).where(GKU.id == uuid.UUID(gku_id))
        )
        gku = result.scalar_one_or_none()
        if not gku:
            return None
        return {
            'id': str(gku.id),
            'workspace_id': str(gku.workspace_id),
            'name': gku.name,
            'description': gku.description,
            'centroid_embedding': gku.centroid_embedding,
            'pattern_signature': gku.pattern_signature,
            'cluster_metadata': gku.cluster_metadata,
            'created_at': gku.created_at
        }

    async def get_gkus_by_workspace(self, workspace_id: str) -> List[Dict[str, Any]]:
        """Get all GKUs for a specific workspace."""
        result = await self.db.execute(
            select(GKU).where(GKU.workspace_id == uuid.UUID(workspace_id))
        )
        gkus = result.scalars().all()
        return [
            {
                'id': str(g.id),
                'workspace_id': str(g.workspace_id),
                'name': g.name,
                'description': g.description,
                'centroid_embedding': g.centroid_embedding,
                'pattern_signature': g.pattern_signature,
                'cluster_metadata': g.cluster_metadata,
                'created_at': g.created_at
            }
            for g in gkus
        ]

    async def associate_akus_with_gku(self, gku_id: str, aku_ids: List[str]):
        """Associate AKUs with a GKU.

        Written through the association Table so ids are stored in the
        dialect's own UUID format (raw SQL used to write dashed strings, which
        SQLite stores differently from the ORM's 32-character hex).
        """
        from sqlalchemy import insert
        gku_uuid = uuid.UUID(str(gku_id))
        existing = set(await self._gku_member_ids(gku_uuid))
        new_rows = []
        for aku_id in aku_ids:
            aku_uuid = uuid.UUID(str(aku_id))
            if aku_uuid not in existing:
                existing.add(aku_uuid)
                new_rows.append({'gku_id': gku_uuid, 'aku_id': aku_uuid})
        if new_rows:
            await self.db.execute(insert(gku_aku_association), new_rows)
        await self.db.commit()

    async def _gku_member_ids(self, gku_uuid: uuid.UUID) -> List[uuid.UUID]:
        """AKU ids linked to a GKU, as UUIDs, including rows written in the
        old dashed-string format (SQLite only)."""
        result = await self.db.execute(
            select(gku_aku_association.c.aku_id).where(gku_aku_association.c.gku_id == gku_uuid)
        )
        ids = {self._as_uuid(v) for (v,) in result.all()}
        if self.db.get_bind().dialect.name == 'sqlite':
            legacy = await self.db.execute(
                text("SELECT aku_id FROM ekm_gku_akus WHERE gku_id = :gku_id"),
                {'gku_id': str(gku_uuid)}
            )
            ids.update(self._as_uuid(v) for (v,) in legacy.all())
        ids.discard(None)
        return list(ids)

    @staticmethod
    def _as_uuid(value) -> Optional[uuid.UUID]:
        if isinstance(value, uuid.UUID):
            return value
        try:
            return uuid.UUID(str(value))
        except (TypeError, ValueError):
            return None

    async def get_akus_in_gku(self, gku_id: str) -> List[Dict[str, Any]]:
        """Active (non-archived) AKUs associated with a GKU."""
        member_ids = await self._gku_member_ids(uuid.UUID(str(gku_id)))
        if not member_ids:
            return []
        result = await self.db.execute(
            select(AKU).where(AKU.id.in_(member_ids), AKU.is_archived == False)
        )
        return [
            {
                'id': str(a.id),
                'content': a.content,
                'embedding': self._deserialize_embedding(a.embedding) if a.embedding is not None else None,
                'aku_metadata': a.aku_metadata,
                'episode_id': str(a.episode_id) if a.episode_id else None,
                'created_at': a.created_at
            }
            for a in result.scalars().all()
        ]

    async def get_gku_ids_for_akus(self, workspace_id: str, aku_ids: List[str]) -> Dict[str, List[str]]:
        """{aku_id: [GKU ids containing it]} for the given AKUs, in one query
        (plus one for rows in the old dashed-string format on SQLite)."""
        uuids = [u for u in (self._as_uuid(i) for i in aku_ids) if u is not None]
        if not uuids:
            return {}
        result = await self.db.execute(
            select(gku_aku_association.c.aku_id, gku_aku_association.c.gku_id)
            .join(GKU, GKU.id == gku_aku_association.c.gku_id)
            .where(gku_aku_association.c.aku_id.in_(uuids), GKU.workspace_id == uuid.UUID(workspace_id))
        )
        pairs = [(self._as_uuid(a), self._as_uuid(g)) for a, g in result.all()]
        if self.db.get_bind().dialect.name == 'sqlite':
            from sqlalchemy import bindparam
            legacy = await self.db.execute(
                text("SELECT aku_id, gku_id FROM ekm_gku_akus WHERE aku_id IN :ids")
                .bindparams(bindparam('ids', expanding=True)),
                {'ids': [str(u) for u in uuids]}
            )
            workspace_gkus = {self._as_uuid(g['id']) for g in await self.get_gkus_by_workspace(workspace_id)}
            pairs += [(self._as_uuid(a), self._as_uuid(g)) for a, g in legacy.all()
                      if self._as_uuid(g) in workspace_gkus]
        membership: Dict[str, List[str]] = {}
        for aku_uuid, gku_uuid in pairs:
            if aku_uuid is None or gku_uuid is None:
                continue
            parents = membership.setdefault(str(aku_uuid), [])
            if str(gku_uuid) not in parents:
                parents.append(str(gku_uuid))
        return membership

    async def get_akus_with_embeddings(self, workspace_id: str) -> List[Dict[str, Any]]:
        """Get all AKUs in a workspace that have embeddings."""
        result = await self.db.execute(
            select(AKU).where(
                and_(
                    AKU.workspace_id == uuid.UUID(workspace_id),
                    AKU.embedding.isnot(None),
                    AKU.is_archived == False
                )
            )
        )
        akus = result.scalars().all()

        return [
            {
                'id': str(aku.id),
                'content': aku.content,
                'embedding': aku.embedding,
                'aku_metadata': aku.aku_metadata,
                'episode_id': str(aku.episode_id) if aku.episode_id else None,
                'created_at': aku.created_at
            }
            for aku in akus
            if aku.embedding is not None  # Double-check that embedding exists
        ]

    async def update_gku_pattern_signature(self, gku_id: str, pattern_signature: Dict[str, Any]):
        """Update the pattern signature of an existing GKU."""
        result = await self.db.execute(
            select(GKU).where(GKU.id == uuid.UUID(gku_id))
        )
        gku = result.scalar_one_or_none()
        if gku:
            gku.pattern_signature = self._standardize_data(pattern_signature)
            await self.db.commit()

    async def save_rl_state(self, model_id: str, weights: List[float], metadata: Optional[Dict[str, Any]] = None):
        """Save RL model weights and state for persistence across restarts."""
        from datetime import datetime
        from sqlalchemy import text

        # Use raw SQL to handle the RL state table (upsert pattern)
        # First, check if a row exists
        result = await self.db.execute(
            text("SELECT id FROM ekm_rl_state WHERE model_id = :model_id"),
            {"model_id": model_id}
        )
        existing = result.fetchone()

        weights_json = json.dumps(self._standardize_data(weights))
        metadata_json = json.dumps(self._standardize_data(metadata)) if metadata else None

        if existing:
            # Update existing
            await self.db.execute(
                text("""
                    UPDATE ekm_rl_state
                    SET weights = :weights, rl_metadata = :metadata, updated_at = :updated_at
                    WHERE model_id = :model_id
                """),
                {
                    "model_id": model_id,
                    "weights": weights_json,
                    "metadata": metadata_json,
                    "updated_at": datetime.utcnow()
                }
            )
        else:
            # Insert new
            new_id = str(uuid.uuid4())
            await self.db.execute(
                text("""
                    INSERT INTO ekm_rl_state (id, model_id, weights, rl_metadata, created_at, updated_at)
                    VALUES (:id, :model_id, :weights, :metadata, :created_at, :updated_at)
                """),
                {
                    "id": new_id,
                    "model_id": model_id,
                    "weights": weights_json,
                    "metadata": metadata_json,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            )

        await self.db.commit()

    async def load_rl_state(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load RL model weights and state."""
        from sqlalchemy import text
        
        result = await self.db.execute(
            text("SELECT weights, rl_metadata, updated_at FROM ekm_rl_state WHERE model_id = :model_id"),
            {"model_id": model_id}
        )
        row = result.fetchone()

        if not row:
            return None

        return {
            "weights": json.loads(row[0]) if row[0] else [],
            "metadata": json.loads(row[1]) if row[1] else {},
            "updated_at": row[2]
        }
