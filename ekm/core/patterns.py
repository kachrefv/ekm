"""
Pattern extraction module for EKM - Implements various pattern recognition techniques
including graphlet distribution, random walks, degree statistics, and semantic centroids.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import Counter
import random

try:
    from scipy.sparse import csr_matrix, lil_matrix
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..storage.base import BaseStorage
from .adaptive_k import suggest_adaptive_k
from .validation import validate_sparse_approximation, compute_approximation_quality

logger = logging.getLogger(__name__)


def count_graphlets(adjacency_matrix: Any, max_nodes: int = 4) -> Dict[str, int]:
    """
    Count occurrences of different graphlets in the adjacency matrix.
    Uses sparse matrix optimizations if scipy is available for O(k) efficiency.
    """
    if not SCIPY_AVAILABLE or not isinstance(adjacency_matrix, (csr_matrix, lil_matrix)):
        # Fallback to dense if necessary, but convert to dense first
        if hasattr(adjacency_matrix, "toarray"):
            adj = adjacency_matrix.toarray()
        else:
            adj = np.array(adjacency_matrix)
        return _count_graphlets_dense(adj)

    # Sparse optimization for 3-node graphlets
    # Ensure symmetric
    adj = adjacency_matrix.tocsr()
    adj = (adj + adj.T) > 0
    adj = adj.astype(np.float32)

    n = adj.shape[0]
    if n < 3:
        return {'triangle': 0, 'wedge': 0, 'triplet': 0}

    # Efficient triangle counting using node-iteration method for sparse graphs
    # This is more efficient than A^3 when the graph is sparse
    triangles = _count_triangles_sparse_efficient(adj)

    # Wedges: Number of paths of length 2 that are NOT triangles
    # Total paths of length 2 = sum(degrees * (degrees - 1) / 2)
    degrees = np.array(adj.sum(axis=0)).flatten()
    total_2_paths = int(np.sum(degrees * (degrees - 1) / 2))
    wedges = total_2_paths - 3 * triangles

    # Triplets: C(n, 3) - triangles - wedges
    total_possible = n * (n - 1) * (n - 2) // 6
    triplets = total_possible - triangles - wedges

    return {
        'triangle': triangles,
        'wedge': wedges,
        'triplet': max(0, triplets)
    }


def _count_triangles_sparse_efficient(adj_matrix):
    """
    Efficient triangle counting for sparse graphs using node-iteration method.
    Complexity: O(V * d_max^2) where d_max is the maximum degree, much better
    than O(V^3) for dense methods when the graph is sparse.
    """
    n = adj_matrix.shape[0]
    triangles = 0

    # Convert to CSR for efficient row access
    adj_csr = adj_matrix.tocsr()

    for i in range(n):
        # Get neighbors of node i
        start_i = adj_csr.indptr[i]
        end_i = adj_csr.indptr[i + 1]
        neighbors_i = adj_csr.indices[start_i:end_i]

        # For each pair of neighbors of i, check if they are connected
        for idx_j, j in enumerate(neighbors_i):
            if j <= i:  # Only consider j > i to avoid double counting
                continue
            start_j = adj_csr.indptr[j]
            end_j = adj_csr.indptr[j + 1]
            neighbors_j = adj_csr.indices[start_j:end_j]

            # Find intersection of neighbors_i and neighbors_j
            # Since both are sorted, we can use a merge-like approach
            idx_k = 0
            for k in neighbors_i[idx_j + 1:]:  # Only consider k > j to avoid double counting
                while idx_k < len(neighbors_j) and neighbors_j[idx_k] < k:
                    idx_k += 1
                if idx_k < len(neighbors_j) and neighbors_j[idx_k] == k:
                    triangles += 1

    return triangles

def _count_graphlets_dense(adj: np.ndarray) -> Dict[str, int]:
    """Dense fallback for graphlet counting."""
    n = adj.shape[0]
    if n < 3:
        return {'triangle': 0, 'wedge': 0, 'triplet': 0}
    
    # Symmetric check
    adj = (adj + adj.T) > 0
    adj = adj.astype(float)
    
    triangle_count = 0
    wedge_count = 0
    
    # Count triplets (i, j, k) with i < j < k
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                edges = 0
                if adj[i, j]: edges += 1
                if adj[i, k]: edges += 1
                if adj[j, k]: edges += 1
                
                if edges == 3:
                    triangle_count += 1
                elif edges == 2:
                    wedge_count += 1
    
    # Each wedge is counted once because exactly one node is the "center" 
    # (the node i connected to both j and k).
    # Each triangle is counted once because i < j < k.
    
    total_triplets = n * (n-1) * (n-2) // 6
    triplet_count = total_triplets - triangle_count - wedge_count
    
    return {
        'triangle': triangle_count,
        'wedge': wedge_count,
        'triplet': max(0, int(triplet_count))
    }


def compute_random_walk_patterns(adjacency_matrix: Any, num_walks: int = 20, walk_length: int = 5) -> Dict[str, float]:
    """
    Compute features based on random walks on the graph.
    Uses sparse matrix powers if available for efficiency.
    """
    if not SCIPY_AVAILABLE or not isinstance(adjacency_matrix, (csr_matrix, lil_matrix)):
        return _compute_random_walk_dense(adjacency_matrix, num_walks, walk_length)

    n = adjacency_matrix.shape[0]
    if n == 0:
        return {'return_probability': 0.0, 'path_diversity': 0.0}
    
    # Row-normalize to get transition matrix
    adj = adjacency_matrix.tocsr()
    row_sums = np.array(adj.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    
    from scipy.sparse import diags
    P = diags(1.0 / row_sums).dot(adj)
    
    # Return probability: average of diagonal of P^k
    # For small walk_length, we can compute P^length efficiently
    Pk = P
    for _ in range(walk_length - 1):
        Pk = Pk.dot(P)
    
    return_prob = float(Pk.diagonal().mean())
    
    # Path diversity: proxy using variance of visit counts
    visit_counts = np.array(Pk.sum(axis=0)).flatten()
    path_diversity = float(np.std(visit_counts))

    return {
        'return_probability': return_prob,
        'path_diversity': path_diversity
    }

def _compute_random_walk_dense(adj_matrix: np.ndarray, num_walks: int, walk_length: int) -> Dict[str, float]:
    """Dense fallback for random walk logic using simulation."""
    if hasattr(adj_matrix, "toarray"):
        adj = adj_matrix.toarray()
    else:
        adj = np.array(adj_matrix)
    
    n = adj.shape[0]
    if n == 0:
        return {'return_probability': 0.0, 'path_diversity': 0.0}
    
    all_walks = []
    return_count = 0
    
    for start_node in range(n):
        for _ in range(num_walks):
            current = start_node
            walk = [current]
            for _ in range(walk_length):
                neighbors = np.where(adj[current] > 0)[0]
                if neighbors.size == 0:
                    break
                current = int(random.choice(neighbors.tolist()))
                walk.append(current)
            
            all_walks.append(walk)
            if len(walk) > 1 and walk[-1] == start_node:
                return_count += 1
    
    return_prob = return_count / (n * num_walks) if n > 0 else 0.0
    unique_paths = len(set(tuple(w) for w in all_walks))
    path_diversity = unique_paths / (n * num_walks) if n > 0 else 0.0
    
    return {
        'return_probability': return_prob,
        'path_diversity': path_diversity
    }


def compute_degree_statistics(adjacency_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Compute degree statistics for the graph represented by the adjacency matrix.
    """
    n = adjacency_matrix.shape[0]
    if n == 0:
        return {
            'mean_degree': 0,
            'std_degree': 0,
            'max_degree': 0,
            'min_degree': 0,
            'degree_distribution': {}
        }
    
    # Calculate degrees (sum of each row for undirected graph)
    # Use np.asarray().flatten() to ensure we have a 1D array even with sparse matrices
    if hasattr(adjacency_matrix, "sum"):
        degrees = np.asarray(adjacency_matrix.sum(axis=1)).flatten()
    else:
        degrees = np.sum(adjacency_matrix, axis=1).flatten()
    
    # Compute statistics
    mean_degree = float(np.mean(degrees))
    std_degree = float(np.std(degrees))
    max_degree = int(np.max(degrees))
    min_degree = int(np.min(degrees))
    
    # Degree distribution
    degree_counts = Counter(degrees)
    degree_distribution = {int(degree): count for degree, count in degree_counts.items()}
    
    return {
        'mean_degree': mean_degree,
        'std_degree': std_degree,
        'max_degree': max_degree,
        'min_degree': min_degree,
        'degree_distribution': degree_distribution
    }


def compute_semantic_centroids(embeddings: List[List[float]], cluster_indices: List[int]) -> Dict[int, List[float]]:
    """
    Compute semantic centroids for different clusters of nodes.
    """
    if not embeddings or not cluster_indices:
        return {}
    
    # Convert to numpy arrays
    embeddings_array = np.array(embeddings)
    clusters = np.array(cluster_indices)
    
    # Get unique clusters
    unique_clusters = np.unique(clusters)
    
    centroids = {}
    for cluster_id in unique_clusters:
        # Get indices of nodes in this cluster
        cluster_mask = (clusters == cluster_id)
        cluster_embeddings = embeddings_array[cluster_mask]
        
        # Compute centroid (mean) of embeddings in this cluster
        centroid = np.mean(cluster_embeddings, axis=0).tolist()
        centroids[int(cluster_id)] = centroid
    
    return centroids


def extract_pattern_signature(
    adjacency_matrix: np.ndarray, 
    embeddings: Optional[List[List[float]]] = None,
    cluster_labels: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Extract comprehensive pattern signature combining all features.
    """
    signature = {}
    
    # 1. Graphlet distribution
    signature['graphlets'] = count_graphlets(adjacency_matrix)
    
    # 2. Random walk patterns
    signature['random_walks'] = compute_random_walk_patterns(adjacency_matrix)
    
    # 3. Degree statistics
    signature['degree_stats'] = compute_degree_statistics(adjacency_matrix)
    
    # 4. Semantic centroids (if embeddings provided)
    if embeddings is not None and cluster_labels is not None:
        signature['semantic_centroids'] = compute_semantic_centroids(embeddings, cluster_labels)
    elif embeddings is not None:
        # If no cluster labels provided, treat all as one cluster
        signature['semantic_centroids'] = compute_semantic_centroids(embeddings, [0] * len(embeddings))
    
    # 5. Additional graph properties
    n = adjacency_matrix.shape[0]
    if n > 0:
        # Density
        total_possible_edges = n * (n - 1)  # assuming directed graph, adjust if undirected
        actual_edges = np.sum(adjacency_matrix)
        density = actual_edges / total_possible_edges if total_possible_edges > 0 else 0.0
        
        signature['graph_properties'] = {
            'density': density,
            'num_nodes': n,
            'num_edges': int(actual_edges)
        }
    else:
        signature['graph_properties'] = {
            'density': 0.0,
            'num_nodes': 0,
            'num_edges': 0
        }
    
    return signature


def calculate_pattern_similarity(sig1: Dict[str, Any], sig2: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate similarity between two pattern signatures using weighted combination of features.
    Optimized for efficiency with early termination and vectorized operations.
    """
    if weights is None:
        weights = {
            'graphlets': 0.3,
            'random_walks': 0.2,
            'degree_stats': 0.2,
            'semantic_centroids': 0.3
        }

    similarities = {}
    total_weight = 0.0
    weighted_sum = 0.0

    # 1. Graphlet similarity (using Jaccard-like coefficient)
    if 'graphlets' in sig1 and 'graphlets' in sig2:
        g1 = sig1['graphlets']
        g2 = sig2['graphlets']
        
        # Use numpy for faster computation
        keys = set(g1.keys()) & set(g2.keys())
        if keys:
            vals1 = np.array([g1[k] for k in keys])
            vals2 = np.array([g2[k] for k in keys])
            
            intersection = np.sum(np.minimum(vals1, vals2))
            union = np.sum(np.maximum(vals1, vals2))
            
            graphlet_sim = intersection / union if union > 0 else 0.0
        else:
            graphlet_sim = 0.0
            
        similarities['graphlets'] = graphlet_sim
        weight = weights.get('graphlets', 0.0)
        if weight > 0:
            weighted_sum += weight * graphlet_sim
            total_weight += weight

    # 2. Random walk similarity (comparing pattern counts and transition probabilities)
    if 'random_walks' in sig1 and 'random_walks' in sig2 and weights.get('random_walks', 0) > 0:
        rw1 = sig1['random_walks']
        rw2 = sig2['random_walks']

        # Compare number of unique patterns
        count1 = rw1.get('num_unique_patterns', 0)
        count2 = rw2.get('num_unique_patterns', 0)
        max_count = max(count1, count2)
        pattern_count_sim = (min(count1, count2) / max_count) if max_count > 0 else 1.0

        # Compare average transition probabilities
        avg_prob1 = rw1.get('avg_transition_prob', 0.0)
        avg_prob2 = rw2.get('avg_transition_prob', 0.0)
        prob_diff = abs(avg_prob1 - avg_prob2)
        transition_prob_sim = max(0.0, 1.0 - prob_diff)

        # Combine both aspects
        random_walk_sim = 0.5 * pattern_count_sim + 0.5 * transition_prob_sim
        similarities['random_walks'] = random_walk_sim
        
        weight = weights.get('random_walks', 0.0)
        if weight > 0:
            weighted_sum += weight * random_walk_sim
            total_weight += weight

    # 3. Degree statistics similarity
    if 'degree_stats' in sig1 and 'degree_stats' in sig2 and weights.get('degree_stats', 0) > 0:
        ds1 = sig1['degree_stats']
        ds2 = sig2['degree_stats']

        # Compare mean degrees (normalized)
        mean_deg1 = ds1.get('mean_degree', 0.0)
        mean_deg2 = ds2.get('mean_degree', 0.0)
        max_mean = max(mean_deg1, mean_deg2, 1.0)  # Avoid division by zero
        mean_sim = 1.0 - abs(mean_deg1 - mean_deg2) / max_mean if max_mean > 0 else 1.0

        # Compare std degrees
        std_deg1 = ds1.get('std_degree', 0.0)
        std_deg2 = ds2.get('std_degree', 0.0)
        max_std = max(std_deg1, std_deg2, 1.0)
        std_sim = 1.0 - abs(std_deg1 - std_deg2) / max_std if max_std > 0 else 1.0

        degree_sim = 0.6 * mean_sim + 0.4 * std_sim
        similarities['degree_stats'] = degree_sim
        
        weight = weights.get('degree_stats', 0.0)
        if weight > 0:
            weighted_sum += weight * degree_sim
            total_weight += weight

    # 4. Semantic centroid similarity (if available)
    if 'semantic_centroids' in sig1 and 'semantic_centroids' in sig2 and weights.get('semantic_centroids', 0) > 0:
        sc1 = sig1['semantic_centroids']
        sc2 = sig2['semantic_centroids']

        if sc1 and sc2:
            # Get matching cluster IDs
            matching_ids = set(sc1.keys()) & set(sc2.keys())
            
            if matching_ids:
                # Convert to numpy arrays for vectorized operations
                emb1_list = []
                emb2_list = []
                
                for cluster_id in matching_ids:
                    emb1_list.append(sc1[cluster_id])
                    emb2_list.append(sc2[cluster_id])
                
                if emb1_list and emb2_list:
                    emb1_array = np.array(emb1_list)
                    emb2_array = np.array(emb2_list)
                    
                    # Compute cosine similarities in bulk
                    dot_products = np.einsum('ij,ij->i', emb1_array, emb2_array)  # Element-wise dot products
                    norms1 = np.linalg.norm(emb1_array, axis=1)
                    norms2 = np.linalg.norm(emb2_array, axis=1)
                    
                    # Avoid division by zero
                    valid_mask = (norms1 > 0) & (norms2 > 0)
                    if np.any(valid_mask):
                        cos_sims = np.zeros_like(dot_products)
                        cos_sims[valid_mask] = dot_products[valid_mask] / (norms1[valid_mask] * norms2[valid_mask])
                        semantic_sim = float(np.mean(cos_sims[valid_mask]))
                    else:
                        semantic_sim = 0.0
                else:
                    semantic_sim = 0.0
            else:
                semantic_sim = 0.0
        else:
            semantic_sim = 1.0 if not sc1 and not sc2 else 0.0  # Both empty = similar, one empty = dissimilar

        similarities['semantic_centroids'] = semantic_sim
        
        weight = weights.get('semantic_centroids', 0.0)
        if weight > 0:
            weighted_sum += weight * semantic_sim
            total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight


async def extract_and_store_patterns_for_gku(
    storage: BaseStorage,
    gku_id: str,
    adjacency_matrix: np.ndarray,
    aku_embeddings: Optional[List[List[float]]] = None,
    cluster_labels: Optional[List[int]] = None
) -> bool:
    """
    Extract pattern signature for a GKU and store it in the database.
    """
    try:
        # Extract pattern signature
        pattern_signature = extract_pattern_signature(
            adjacency_matrix=adjacency_matrix,
            embeddings=aku_embeddings,
            cluster_labels=cluster_labels
        )
        
        # Update the GKU with the pattern signature
        await storage.update_gku_pattern_signature(gku_id, pattern_signature)
        
        return True
    except Exception as e:
        logger.error(f"Failed to extract and store patterns for GKU {gku_id}: {e}")
        return False