import numpy as np
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SparseGraphManager:
    """Manages sparse adjacency matrix for spreading activation."""

    def __init__(self, semantic_weight: float = 0.5, temporal_weight: float = 0.3, causal_weight: float = 0.2):
        try:
            from scipy.sparse import lil_matrix, csr_matrix
            self.lil_matrix = lil_matrix
            self.csr_matrix = csr_matrix
            self.scipy_available = True
        except ImportError:
            self.scipy_available = False
            logger.warning("scipy not installed. Using dense matrix fallback.")

        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.adjacency: Optional[np.ndarray] = None
        self.current_size = 0

        total_weight = semantic_weight + temporal_weight + causal_weight
        self.semantic_weight = semantic_weight / total_weight if total_weight > 0 else 0.5
        self.temporal_weight = temporal_weight / total_weight if total_weight > 0 else 0.3
        self.causal_weight = causal_weight / total_weight if total_weight > 0 else 0.2

    def add_nodes(self, ids: List[str]) -> None:
        for node_id in ids:
            if node_id not in self.id_to_idx:
                idx = self.current_size
                self.id_to_idx[node_id] = idx
                self.idx_to_id[idx] = node_id
                self.current_size += 1

    def add_edges(self, relationships: List[Dict]) -> None:
        if not relationships:
            return

        all_ids = set()
        for rel in relationships:
            all_ids.add(rel['source_aku_id'])
            all_ids.add(rel['target_aku_id'])

        new_ids = [id for id in all_ids if id not in self.id_to_idx]
        self.add_nodes(new_ids)

        n = self.current_size
        if self.scipy_available:
            if self.adjacency is None:
                self.adjacency = self.lil_matrix((n, n), dtype=np.float32)
            elif self.adjacency.shape[0] < n:
                old = self.adjacency.tocsr()
                self.adjacency = self.lil_matrix((n, n), dtype=np.float32)
                self.adjacency[:old.shape[0], :old.shape[1]] = old
        else:
            if self.adjacency is None:
                self.adjacency = np.zeros((n, n), dtype=np.float32)
            elif self.adjacency.shape[0] < n:
                new_adj = np.zeros((n, n), dtype=np.float32)
                new_adj[:self.adjacency.shape[0], :self.adjacency.shape[1]] = self.adjacency
                self.adjacency = new_adj

        for rel in relationships:
            src_idx = self.id_to_idx.get(rel['source_aku_id'])
            tgt_idx = self.id_to_idx.get(rel['target_aku_id'])
            if src_idx is not None and tgt_idx is not None:
                weight = (
                    self.semantic_weight * rel.get('semantic_similarity', 0) +
                    self.temporal_weight * rel.get('temporal_proximity', 0) +
                    self.causal_weight * rel.get('causal_weight', 0)
                )
                self.adjacency[src_idx, tgt_idx] = weight
                # Add weak reverse edge
                self.adjacency[tgt_idx, src_idx] = weight * 0.1

    def spreading_activation(
        self,
        initial_scores: Dict[str, float],
        iterations: int = 3,
        decay: float = 0.5,
        mode: str = "fixed"
    ) -> Dict[str, Any]:
        if self.adjacency is None or self.current_size == 0:
            return {'scores': initial_scores}

        n = self.current_size
        activation = np.zeros(n, dtype=np.float32)
        for node_id, score in initial_scores.items():
            if node_id in self.id_to_idx:
                activation[self.id_to_idx[node_id]] = score

        if self.scipy_available:
            adj = self.adjacency.tocsr()
        else:
            adj = self.adjacency

        for _ in range(iterations):
            spread = adj.T @ activation
            activation = activation + decay * spread
            activation = np.clip(activation, 0, 2.0)

        scores = {}
        for idx in range(n):
            if activation[idx] > 0.01:
                scores[self.idx_to_id[idx]] = float(activation[idx])

        return {'scores': scores}
