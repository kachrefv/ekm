import uuid
import json
import logging
import numpy as np
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ekm.core.models import AKU, GKU, AKURelationship, gku_aku_association
from ..core.dependencies import get_ekm_instance, get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Graph"])

def _deserialize_embedding(emb):
    """Safely deserialize an embedding to a list of floats."""
    if emb is None:
        return None
    if isinstance(emb, str):
        try:
            return json.loads(emb)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(emb, (bytes, bytearray)):
        try:
            return json.loads(emb.decode("utf-8"))
        except:
            return None
    if isinstance(emb, (list, tuple)):
        return list(emb)
    if hasattr(emb, "tolist"):
        return emb.tolist()
    return None


def _project_to_3d(embeddings: List[List[float]], scale: float = 8.0) -> List[List[float]]:
    """Project high-dimensional embeddings to 3D using PCA. Falls back to random."""
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [[0.0, 0.0, 0.0]]

    mat = np.array(embeddings, dtype=np.float32)
    std = mat.std(axis=0)
    valid = std > 1e-8
    if valid.sum() < 3:
        rng = np.random.default_rng(42)
        return (rng.standard_normal((n, 3)) * scale).tolist()

    mat = mat[:, valid]
    mat = mat - mat.mean(axis=0)

    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(3, mat.shape[1]), random_state=42)
        projected = pca.fit_transform(mat)
    except ImportError:
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        projected = U[:, :3] * S[:3]

    if projected.shape[1] < 3:
        pad = np.zeros((n, 3 - projected.shape[1]))
        projected = np.hstack([projected, pad])

    max_abs = np.abs(projected).max()
    if max_abs > 0:
        projected = projected / max_abs * scale

    return projected.tolist()


@router.get("/graph/{workspace_id}")
async def get_graph(workspace_id: str, db: Session = Depends(get_db)):
    """Return the full knowledge graph for 3D visualization."""
    # Ensure EKM is initialized (might resolve dependencies/config)
    get_ekm_instance(workspace_id)
    
    try:
        ws_uuid = uuid.UUID(workspace_id)

        # --- Fetch AKUs ---
        akus = db.query(AKU).filter(AKU.workspace_id == ws_uuid).all()
        aku_embeddings = []
        aku_ids = []
        aku_has_embedding = {}
        for a in akus:
            emb = _deserialize_embedding(a.embedding)
            if emb and len(emb) > 0:
                aku_embeddings.append(emb)
                aku_has_embedding[str(a.id)] = len(aku_embeddings) - 1
            aku_ids.append(str(a.id))

        # --- Fetch GKUs ---
        gkus = db.query(GKU).filter(GKU.workspace_id == ws_uuid).all()
        gku_embeddings = []
        gku_has_embedding = {}
        for g in gkus:
            emb = _deserialize_embedding(g.centroid_embedding)
            if emb and len(emb) > 0:
                gku_embeddings.append(emb)
                gku_has_embedding[str(g.id)] = len(gku_embeddings) - 1

        # --- Project to 3D ---
        all_embeddings = aku_embeddings + gku_embeddings
        positions_3d = _project_to_3d(all_embeddings)

        rng = np.random.default_rng(7)

        nodes = []
        for a in akus:
            aid = str(a.id)
            if aid in aku_has_embedding:
                idx = aku_has_embedding[aid]
                pos = positions_3d[idx]
            else:
                pos = (rng.standard_normal(3) * 6).tolist()
            nodes.append({
                "id": aid,
                "type": "aku",
                "label": (a.content or "")[:80],
                "content": a.content or "",
                "x": round(pos[0], 3),
                "y": round(pos[1], 3),
                "z": round(pos[2], 3),
                "archived": bool(a.is_archived),
            })

        gku_offset = len(aku_embeddings)
        for g in gkus:
            gid = str(g.id)
            if gid in gku_has_embedding:
                idx = gku_offset + gku_has_embedding[gid]
                pos = positions_3d[idx]
            else:
                pos = (rng.standard_normal(3) * 4).tolist()
            nodes.append({
                "id": gid,
                "type": "gku",
                "label": g.name or "Unnamed Concept",
                "content": g.name or "",
                "x": round(pos[0], 3),
                "y": round(pos[1], 3),
                "z": round(pos[2], 3),
                "pattern": g.pattern_signature,
                "member_count": len(g.member_akus) if hasattr(g, 'member_akus') else 0,
            })

        # --- Fetch edges (AKU relationships) ---
        rels = db.query(AKURelationship).filter(AKURelationship.workspace_id == ws_uuid).all()
        edges = []
        for r in rels:
            edge_type = "semantic"
            weight = r.semantic_similarity or 0.0
            if (r.causal_weight or 0) > weight:
                edge_type = "causal"
                weight = r.causal_weight
            if (r.temporal_proximity or 0) > weight:
                edge_type = "temporal"
                weight = r.temporal_proximity
            edges.append({
                "source": str(r.source_aku_id),
                "target": str(r.target_aku_id),
                "weight": round(weight, 3),
                "type": edge_type,
                "semantic": round(r.semantic_similarity or 0, 3),
                "causal": round(r.causal_weight or 0, 3),
                "temporal": round(r.temporal_proximity or 0, 3),
            })

        # --- GKU → AKU membership edges ---
        assoc_rows = db.query(gku_aku_association).all()
        for row in assoc_rows:
            edges.append({
                "source": str(row.gku_id),
                "target": str(row.aku_id),
                "weight": 1.0,
                "type": "membership",
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "aku_count": len(akus),
                "gku_count": len(gkus),
                "edge_count": len(edges),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
