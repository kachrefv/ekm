"""
Adaptive K Selection Module for EKM - Dynamically adjusts the k value for k-NN
approximation based on graph density and other structural properties.
"""

import numpy as np
from typing import Union, Tuple
from scipy.sparse import csr_matrix, issparse


def compute_graph_density(adjacency_matrix: Union[np.ndarray, csr_matrix]) -> float:
    """
    Compute the density of the graph represented by the adjacency matrix.
    
    Args:
        adjacency_matrix: The adjacency matrix (dense or sparse)
        
    Returns:
        Graph density as a float between 0 and 1
    """
    if issparse(adjacency_matrix):
        n = adjacency_matrix.shape[0]
        if n <= 1:
            return 0.0
        
        # For undirected graphs, count each edge twice (i->j and j->i)
        # but divide by total possible directed connections
        edges = adjacency_matrix.nnz
        total_possible = n * (n - 1)  # Directed graph
        return edges / total_possible if total_possible > 0 else 0.0
    else:
        n = adjacency_matrix.shape[0]
        if n <= 1:
            return 0.0
            
        edges = np.count_nonzero(adjacency_matrix)
        total_possible = n * (n - 1)  # Directed graph
        return edges / total_possible if total_possible > 0 else 0.0


def compute_local_density(adjacency_matrix: Union[np.ndarray, csr_matrix], node_idx: int) -> float:
    """
    Compute the local density around a specific node.
    
    Args:
        adjacency_matrix: The adjacency matrix
        node_idx: Index of the node to compute local density for
        
    Returns:
        Local density around the node
    """
    if issparse(adjacency_matrix):
        # Get neighbors of the node
        start_idx = adjacency_matrix.indptr[node_idx]
        end_idx = adjacency_matrix.indptr[node_idx + 1]
        neighbors = adjacency_matrix.indices[start_idx:end_idx]
        
        if len(neighbors) < 2:
            return 0.0
            
        # Count connections among neighbors
        connected_pairs = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if adjacency_matrix[neighbors[i], neighbors[j]] != 0:
                    connected_pairs += 1
                    
        # Maximum possible connections among neighbors
        max_connections = len(neighbors) * (len(neighbors) - 1) / 2
        return connected_pairs / max_connections if max_connections > 0 else 0.0
    else:
        n = adjacency_matrix.shape[0]
        neighbors = np.where(adjacency_matrix[node_idx, :] != 0)[0]
        
        if len(neighbors) < 2:
            return 0.0
            
        # Count connections among neighbors
        connected_pairs = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if adjacency_matrix[neighbors[i], neighbors[j]] != 0:
                    connected_pairs += 1
                    
        # Maximum possible connections among neighbors
        max_connections = len(neighbors) * (len(neighbors) - 1) / 2
        return connected_pairs / max_connections if max_connections > 0 else 0.0


def compute_clustering_coefficient(adjacency_matrix: Union[np.ndarray, csr_matrix]) -> float:
    """
    Compute the average clustering coefficient of the graph.
    
    Args:
        adjacency_matrix: The adjacency matrix
        
    Returns:
        Average clustering coefficient
    """
    n = adjacency_matrix.shape[0]
    if n < 3:
        return 0.0
        
    total_cc = 0.0
    count = 0
    
    for i in range(n):
        local_cc = compute_local_density(adjacency_matrix, i)
        if local_cc > 0:  # Only count nodes that have neighbors
            total_cc += local_cc
            count += 1
            
    return total_cc / count if count > 0 else 0.0


def suggest_adaptive_k(
    adjacency_matrix: Union[np.ndarray, csr_matrix], 
    base_k: int = 20,
    min_k: int = 5,
    max_k: int = 100
) -> Tuple[int, dict]:
    """
    Suggest an adaptive k value based on graph properties.
    
    Args:
        adjacency_matrix: The adjacency matrix
        base_k: Base k value to adjust from
        min_k: Minimum allowed k value
        max_k: Maximum allowed k value
        
    Returns:
        Tuple of (adaptive_k, metadata_dict)
    """
    n = adjacency_matrix.shape[0]
    if n == 0:
        return min_k, {"reason": "empty_graph", "density": 0.0, "clustering_coefficient": 0.0}
    
    # Compute graph properties
    density = compute_graph_density(adjacency_matrix)
    clustering_coeff = compute_clustering_coefficient(adjacency_matrix)
    
    # Adjust k based on graph density
    # Higher density might need more neighbors to capture structure
    density_factor = 1.0 + (density - 0.5) * 0.5  # Adjust by ±25% based on density
    
    # Adjust k based on clustering coefficient
    # Higher clustering might need fewer neighbors due to redundancy
    clustering_factor = 1.0 - clustering_coeff * 0.3  # Reduce by up to 30% for high clustering
    
    # Adjust k based on graph size
    # Larger graphs might need proportionally fewer neighbors
    size_factor = min(1.0, 20.0 / max(n, 1)) * 5.0  # Scale based on graph size
    
    adaptive_k = int(base_k * density_factor * clustering_factor * size_factor)
    
    # Ensure k is within bounds and not more than the number of nodes
    adaptive_k = max(min_k, min(max_k, adaptive_k, n - 1 if n > 1 else min_k))
    
    metadata = {
        "original_k": base_k,
        "density": density,
        "clustering_coefficient": clustering_coeff,
        "density_factor": density_factor,
        "clustering_factor": clustering_factor,
        "size_factor": size_factor,
        "adjusted_k": adaptive_k
    }
    
    return adaptive_k, metadata


def compute_optimal_k_range(
    adjacency_matrix: Union[np.ndarray, csr_matrix],
    min_candidates: int = 5,
    max_candidates: int = 50
) -> Tuple[list, dict]:
    """
    Compute a range of candidate k values to evaluate for optimal performance.
    
    Args:
        adjacency_matrix: The adjacency matrix
        min_candidates: Minimum number of k candidates to generate
        max_candidates: Maximum number of k candidates to generate
        
    Returns:
        Tuple of (k_candidates_list, metadata_dict)
    """
    n = adjacency_matrix.shape[0]
    if n <= 1:
        return [1], {"reason": "too_small"}
    
    # Compute graph properties
    density = compute_graph_density(adjacency_matrix)
    clustering_coeff = compute_clustering_coefficient(adjacency_matrix)
    
    # Generate k candidates based on graph properties
    candidates = set()
    
    # Base recommendations
    candidates.add(max(1, min(n-1, 5)))   # Very small k for initial exploration
    candidates.add(max(1, min(n-1, 10)))  # Small k
    candidates.add(max(1, min(n-1, 20)))  # Medium k
    candidates.add(max(1, min(n-1, 50)))  # Large k
    
    # Density-based recommendations
    if density < 0.1:  # Sparse graph
        candidates.add(max(1, min(n-1, int(10 * (1 - density)))))
    elif density > 0.5:  # Dense graph
        candidates.add(max(1, min(n-1, int(30 * density))))
    else:  # Medium density
        candidates.add(max(1, min(n-1, 15)))
    
    # Clustering-based adjustments
    if clustering_coeff > 0.5:  # High clustering
        # Reduce k due to redundancy in clustered graphs
        candidates.add(max(1, min(n-1, int(15 * (1 - clustering_coeff)))))
    else:
        candidates.add(max(1, min(n-1, int(20 * (1 + clustering_coeff)))))
    
    # Size-based recommendations
    if n > 1000:
        # For large graphs, use logarithmic scaling
        candidates.add(max(1, min(n-1, int(np.log(n) * 10))))
    elif n > 100:
        candidates.add(max(1, min(n-1, int(np.sqrt(n) * 2))))
    
    # Convert to sorted list
    k_candidates = sorted(list(candidates))
    
    # Limit the number of candidates
    if len(k_candidates) > max_candidates:
        # Take evenly spaced candidates
        step = len(k_candidates) // max_candidates
        k_candidates = k_candidates[::step][:max_candidates]
    elif len(k_candidates) < min_candidates:
        # Add intermediate values
        while len(k_candidates) < min_candidates:
            for i in range(len(k_candidates)-1):
                mid_k = (k_candidates[i] + k_candidates[i+1]) // 2
                if mid_k not in k_candidates and mid_k < n:
                    k_candidates.append(mid_k)
                    k_candidates.sort()
                    if len(k_candidates) >= min_candidates:
                        break
    
    metadata = {
        "graph_size": n,
        "density": density,
        "clustering_coefficient": clustering_coeff,
        "initial_candidates": len(candidates),
        "final_candidates": len(k_candidates)
    }
    
    return k_candidates, metadata