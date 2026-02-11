from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, text
import uuid
import json
import numpy as np
from .base import BaseStorage
from ..core.models import Workspace, Episode, AKU, AKURelationship, GKU
from ..core.utils import cosine_similarity

# Import pgvector if available
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False

class SQLStorage(BaseStorage):
    def __init__(self, db: Session):
        self.db = db
        from ..core.scalability import VectorIndexManager
        self.vector_index = VectorIndexManager(dimension=None)
        self._index_populated = False

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
             except:
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
        workspace = self.db.query(Workspace).filter(Workspace.id == uuid.UUID(workspace_id)).first()
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
        self.db.commit()
        
        # Update vector index if it's being used
        if not PGVECTOR_AVAILABLE or not (self.db.bind.dialect.name == 'postgresql'):
             try:
                 emb_array = np.array([embedding]).astype('float32')
                 self.vector_index.add_vectors(emb_array, [str(episode.id)])
             except:
                 pass # Warning: Index update failed
                 
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
        self.db.commit()
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
        self.db.commit()

    async def find_similar_episodes(self, workspace_id: str, embedding: List[float], threshold: float, limit: int) -> List[Dict[str, Any]]:
        query_emb = self._serialize_embedding(embedding)
        
        is_postgres = self.db.bind.dialect.name == 'postgresql'
        
        if PGVECTOR_AVAILABLE and is_postgres:
            # Use pgvector for similarity search
            embedding_array = np.array(query_emb).astype(np.float32).tolist()
            
            # Using raw SQL for pgvector cosine similarity operator
            sql_query = text("""
                SELECT id, content, summary, 
                       (embedding <=> :query_embedding) AS distance
                FROM ekm_episodes 
                WHERE workspace_id = :workspace_id
                ORDER BY distance ASC
                LIMIT :limit
            """)
            
            results = self.db.execute(sql_query, {
                'query_embedding': json.dumps(embedding_array),
                'workspace_id': uuid.UUID(workspace_id),
                'limit': limit
            }).fetchall()
            
            # Convert distance to similarity (1 / (1 + distance))
            return [
                {
                    'id': str(row[0]),
                    'content': row[1],
                    'summary': row[2],
                    'similarity': 1 / (1 + row[3])
                }
                for row in results
            ]
        else:
            # Fallback: using VectorIndexManager (FAISS/Memory)
            
            # Populate index if not already done (Lazy Loading)
            if not self._index_populated:
                # Load all episodes for this workspace
                all_episodes = self.db.query(Episode).filter(Episode.workspace_id == uuid.UUID(workspace_id)).all()
                if all_episodes:
                    vectors = []
                    ids = []
                    for ep in all_episodes:
                        if ep.embedding is not None:
                             emb = ep.embedding
                             # Handle list or json string
                             if isinstance(emb, str):
                                 emb = json.loads(emb)
                             vectors.append(emb)
                             ids.append(str(ep.id))
                    
                    if vectors:
                        self.vector_index.add_vectors(np.array(vectors), ids)
                
                self._index_populated = True
            
            # Search index
            query_vec = np.array(query_emb).astype(np.float32)
            distances, ids = self.vector_index.search(query_vec, limit)
            
            # Fetch full details for the IDs
            results = []
            valid_ids = [id_ for id_ in ids if id_ is not None]
            
            if not valid_ids:
                return []
                
            uuids = [uuid.UUID(i) for i in valid_ids]
            episodes = self.db.query(Episode).filter(Episode.id.in_(uuids)).all()
            episode_map = {str(ep.id): ep for ep in episodes}
            
            for id_, dist in zip(ids, distances):
                if id_ in episode_map:
                    ep = episode_map[id_]
                    sim = 1 / (1 + dist) # Convert distance to similarity
                    if sim >= threshold:
                        results.append({
                            'id': str(ep.id),
                            'summary': ep.summary,
                            'content': ep.content,
                            'similarity': sim
                        })
                        
            return results

    async def find_similar_akus(self, workspace_id: str, embedding: List[float], threshold: float, limit: int) -> List[Dict[str, Any]]:
        query_emb = self._serialize_embedding(embedding)
        
        is_postgres = self.db.bind.dialect.name == 'postgresql'
        
        if PGVECTOR_AVAILABLE and is_postgres:
            # Use pgvector for similarity search
            embedding_array = np.array(query_emb).astype(np.float32).tolist()
            
            # Using raw SQL for pgvector cosine similarity operator
            sql_query = text("""
                SELECT id, content, episode_id,
                       (embedding <=> :query_embedding) AS distance
                FROM ekm_akus 
                WHERE workspace_id = :workspace_id
                  AND is_archived = FALSE
                ORDER BY distance ASC
                LIMIT :limit
            """)
            
            results = self.db.execute(sql_query, {
                'query_embedding': json.dumps(embedding_array),
                'workspace_id': uuid.UUID(workspace_id),
                'limit': limit
            }).fetchall()
            
            # Convert distance to similarity (1 / (1 + distance))
            return [
                {
                    'id': str(row[0]),
                    'content': row[1],
                    'episode_id': str(row[2]) if row[2] else None,
                    'similarity': 1 / (1 + row[3])
                }
                for row in results
            ]
        else:
            # Fallback to original method
            akus = self.db.query(AKU).filter(AKU.workspace_id == uuid.UUID(workspace_id), AKU.is_archived == False).all()
            results = []
            for aku in akus:
                if aku.embedding is not None:
                    sim = cosine_similarity(query_emb, self._serialize_embedding(aku.embedding))
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
        rels = self.db.query(AKURelationship).filter(
            and_(
                or_(AKURelationship.source_aku_id.in_(uuids), AKURelationship.target_aku_id.in_(uuids)),
                AKURelationship.workspace_id == uuid.UUID(workspace_id)
            )
        ).all()
        return [{
            'source_aku_id': str(r.source_aku_id),
            'target_aku_id': str(r.target_aku_id),
            'semantic_similarity': r.semantic_similarity,
            'temporal_proximity': r.temporal_proximity,
            'causal_weight': r.causal_weight
        } for r in rels]

    async def get_akus_by_ids(self, aku_ids: List[str]) -> List[Dict[str, Any]]:
        uuids = [uuid.UUID(i) for i in aku_ids]
        akus = self.db.query(AKU).filter(AKU.id.in_(uuids), AKU.is_archived == False).all()
        return [{
            'id': str(a.id),
            'content': a.content,
            'embedding': a.embedding,
            'aku_metadata': a.aku_metadata,
            'episode_id': str(a.episode_id) if a.episode_id else None
        } for a in akus]

    async def archive_akus(self, aku_ids: List[str]):
        """Mark AKUs as archived."""
        uuids = [uuid.UUID(i) for i in aku_ids]
        self.db.query(AKU).filter(AKU.id.in_(uuids)).update({AKU.is_archived: True}, synchronize_session=False)
        self.db.commit()

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
        self.db.commit()
        return str(gku.id)

    async def get_gku(self, gku_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific GKU by ID."""
        gku = self.db.query(GKU).filter(GKU.id == uuid.UUID(gku_id)).first()
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
        gkus = self.db.query(GKU).filter(GKU.workspace_id == uuid.UUID(workspace_id)).all()
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
        """Associate AKUs with a GKU."""
        from sqlalchemy import text
        
        gku_uuid = uuid.UUID(gku_id)
        aku_uuids = [uuid.UUID(id) for id in aku_ids]
        
        for aku_uuid in aku_uuids:
            # Check if association already exists
            existing = self.db.execute(
                text("SELECT 1 FROM ekm_gku_akus WHERE gku_id = :gku_id AND aku_id = :aku_id"),
                {"gku_id": str(gku_uuid), "aku_id": str(aku_uuid)}
            ).fetchone()
            
            if not existing:
                self.db.execute(
                    text("INSERT INTO ekm_gku_akus (gku_id, aku_id) VALUES (:gku_id, :aku_id)"),
                    {"gku_id": str(gku_uuid), "aku_id": str(aku_uuid)}
                )
        
        self.db.commit()

    async def get_akus_in_gku(self, gku_id: str) -> List[Dict[str, Any]]:
        """Get all AKUs associated with a specific GKU."""
        from sqlalchemy import text
        
        param_uuid = str(uuid.UUID(gku_id))
        
        results = self.db.execute(
            text("""
            SELECT a.id, a.content, a.embedding, a.aku_metadata, a.created_at
            FROM ekm_akus a
            JOIN ekm_gku_akus ga ON a.id = REPLACE(ga.aku_id, '-', '')
            WHERE ga.gku_id = :gku_id
            """),
            {"gku_id": param_uuid}
        ).fetchall()
        
        return [
            {
                'id': str(row[0]),
                'content': row[1],
                'embedding': self._deserialize_embedding(row[2]) if row[2] else None,
                'aku_metadata': row[3],
                'created_at': row[4]
            }
            for row in results
        ]

    async def get_akus_with_embeddings(self, workspace_id: str) -> List[Dict[str, Any]]:
        """Get all AKUs in a workspace that have embeddings."""
        akus = self.db.query(AKU).filter(
            and_(
                AKU.workspace_id == uuid.UUID(workspace_id),
                AKU.embedding.isnot(None),
                AKU.is_archived == False
            )
        ).all()

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
        gku = self.db.query(GKU).filter(GKU.id == uuid.UUID(gku_id)).first()
        if gku:
            gku.pattern_signature = self._standardize_data(pattern_signature)
            self.db.commit()

    async def save_rl_state(self, model_id: str, weights: List[float], metadata: Optional[Dict[str, Any]] = None):
        """Save RL model weights and state for persistence across restarts."""
        from datetime import datetime
        
        # Use raw SQL to handle the RL state table (upsert pattern)
        # First, check if a row exists
        existing = self.db.execute(
            text("SELECT id FROM ekm_rl_state WHERE model_id = :model_id"),
            {"model_id": model_id}
        ).fetchone()
        
        weights_json = json.dumps(self._standardize_data(weights))
        metadata_json = json.dumps(self._standardize_data(metadata)) if metadata else None
        
        if existing:
            # Update existing
            self.db.execute(
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
            self.db.execute(
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
        
        self.db.commit()

    async def load_rl_state(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load RL model weights and state."""
        result = self.db.execute(
            text("SELECT weights, rl_metadata, updated_at FROM ekm_rl_state WHERE model_id = :model_id"),
            {"model_id": model_id}
        ).fetchone()
        
        if not result:
            return None
        
        return {
            "weights": json.loads(result[0]) if result[0] else [],
            "metadata": json.loads(result[1]) if result[1] else {},
            "updated_at": result[2]
        }
