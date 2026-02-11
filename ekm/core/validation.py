"""
Validation module for EKM - Validates that sparse approximations preserve essential graph properties.
"""

import numpy as np
from typing import Dict, Any, Union, Tuple
from scipy.sparse import csr_matrix, issparse
import logging

logger = logging.getLogger(__name__)


def compute_graph_properties(adjacency_matrix: Union[np.ndarray, csr_matrix]) -> Dict[str, Any]:
    """
    Compute essential graph properties for validation.
    
    Args:
        adjacency_matrix: The adjacency matrix (dense or sparse)
        
    Returns:
        Dictionary of graph properties
    """
    if issparse(adjacency_matrix):
        n = adjacency_matrix.shape[0]
        m = adjacency_matrix.nnz
        adj_dense = adjacency_matrix.toarray()
    else:
        n = adjacency_matrix.shape[0]
        m = np.count_nonzero(adjacency_matrix)
        adj_dense = adjacency_matrix.copy()
    
    properties = {
        'num_nodes': n,
        'num_edges': m,
        'density': m / (n * (n - 1)) if n > 1 else 0.0,  # Directed graph
    }
    
    # Compute degree statistics
    if n > 0:
        degrees = np.array(adj_dense.sum(axis=1)).flatten()
        properties.update({
            'mean_degree': float(np.mean(degrees)),
            'std_degree': float(np.std(degrees)),
            'max_degree': int(np.max(degrees)),
            'min_degree': int(np.min(degrees)),
            'degree_assortativity': compute_assortativity(adj_dense)
        })
    
    # Compute clustering coefficient
    properties['clustering_coefficient'] = compute_clustering_coefficient(adj_dense)
    
    # Compute connected components
    properties['num_connected_components'] = compute_num_connected_components(adj_dense)
    
    # Compute diameter (approximate for large graphs)
    properties['diameter'] = compute_diameter_approx(adj_dense)
    
    # Compute spectral properties
    if n <= 1000:  # Only for reasonably sized graphs
        properties.update(compute_spectral_properties(adj_dense))
    else:
        properties.update({
            'largest_eigenvalue': estimate_largest_eigenvalue(adj_dense),
            'spectral_gap': None  # Skip for large graphs
        })
    
    return properties


def compute_assortativity(adj_matrix: np.ndarray) -> float:
    """
    Compute degree assortativity coefficient.
    """
    n = adj_matrix.shape[0]
    if n < 2:
        return 0.0
    
    degrees = adj_matrix.sum(axis=0)  # Out-degrees
    degree_correlation = 0.0
    total_edges = np.sum(adj_matrix)
    
    if total_edges == 0:
        return 0.0
    
    # Calculate correlation between degrees of connected nodes
    numerator = 0.0
    denominator = 0.0
    
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] != 0:
                deg_i = degrees[i]
                deg_j = degrees[j]
                numerator += deg_i * deg_j
    
    # Calculate expected value
    expected_deg_sq = np.sum(degrees ** 2) / n
    expected_deg = np.sum(degrees) / n
    denominator = total_edges * (expected_deg_sq - expected_deg ** 2)
    
    if denominator == 0:
        return 0.0
    
    return (numerator / total_edges - expected_deg ** 2) / (expected_deg_sq - expected_deg ** 2)


def compute_clustering_coefficient(adj_matrix: np.ndarray) -> float:
    """
    Compute average clustering coefficient.
    """
    n = adj_matrix.shape[0]
    if n < 3:
        return 0.0
    
    clustering_coeffs = []
    
    for i in range(n):
        neighbors = np.where(adj_matrix[i, :] != 0)[0]
        if len(neighbors) < 2:
            clustering_coeffs.append(0.0)
            continue
        
        # Count connections between neighbors
        connected_pairs = 0
        for j_idx, j in enumerate(neighbors):
            for k in neighbors[j_idx + 1:]:
                if adj_matrix[j, k] != 0:
                    connected_pairs += 1
        
        # Max possible connections between neighbors
        max_connections = len(neighbors) * (len(neighbors) - 1) / 2
        cc = (2 * connected_pairs) / max_connections if max_connections > 0 else 0.0
        clustering_coeffs.append(cc)
    
    return float(np.mean(clustering_coeffs)) if clustering_coeffs else 0.0


def compute_num_connected_components(adj_matrix: np.ndarray) -> int:
    """
    Compute number of connected components using BFS.
    """
    n = adj_matrix.shape[0]
    if n == 0:
        return 0
    
    visited = np.zeros(n, dtype=bool)
    components = 0
    
    for i in range(n):
        if not visited[i]:
            # BFS to mark all nodes in this component
            queue = [i]
            visited[i] = True
            
            while queue:
                node = queue.pop(0)
                
                # Find all neighbors (both incoming and outgoing for undirected)
                neighbors_out = np.where(adj_matrix[node, :] != 0)[0]
                neighbors_in = np.where(adj_matrix[:, node] != 0)[0]
                all_neighbors = set(neighbors_out).union(set(neighbors_in))
                
                for neighbor in all_neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            components += 1
    
    return components


def compute_diameter_approx(adj_matrix: np.ndarray, max_iterations: int = 10) -> Union[int, float]:
    """
    Approximate graph diameter using random sampling.
    """
    n = adj_matrix.shape[0]
    if n <= 1:
        return 0 if n == 1 else float('inf')
    
    # For small graphs, compute exact diameter
    if n <= 100:
        return compute_exact_diameter(adj_matrix)
    
    # For larger graphs, use approximation
    max_distance = 0
    
    # Sample random nodes to estimate diameter
    for _ in range(min(max_iterations, n)):
        start_node = np.random.randint(0, n)
        distances = bfs_distances(adj_matrix, start_node)
        max_dist = np.max(distances[distances != np.inf])
        max_distance = max(max_distance, max_dist)
    
    return max_distance if max_distance > 0 else float('inf')


def compute_exact_diameter(adj_matrix: np.ndarray) -> Union[int, float]:
    """
    Compute exact diameter using BFS from all nodes (for small graphs).
    """
    n = adj_matrix.shape[0]
    if n <= 1:
        return 0 if n == 1 else float('inf')
    
    max_distance = 0
    
    for i in range(n):
        distances = bfs_distances(adj_matrix, i)
        max_dist = np.max(distances[distances != np.inf])
        max_distance = max(max_distance, max_dist)
    
    return max_distance if max_distance > 0 else float('inf')


def bfs_distances(adj_matrix: np.ndarray, start_node: int) -> np.ndarray:
    """
    Compute shortest path distances from start_node using BFS.
    """
    n = adj_matrix.shape[0]
    distances = np.full(n, np.inf)
    distances[start_node] = 0
    queue = [start_node]
    
    while queue:
        node = queue.pop(0)
        neighbors = np.where(adj_matrix[node, :] != 0)[0]
        
        for neighbor in neighbors:
            if distances[neighbor] == np.inf:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    
    return distances


def compute_spectral_properties(adj_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute spectral properties of the adjacency matrix.
    """
    try:
        eigenvals = np.linalg.eigvals(adj_matrix)
        eigenvals_real = np.real(eigenvals)
        
        largest_eigenval = float(np.max(np.abs(eigenvals_real)))
        
        # Spectral gap (difference between largest and second largest eigenvalues)
        sorted_eigenvals = np.sort(np.abs(eigenvals_real))[::-1]
        if len(sorted_eigenvals) > 1:
            spectral_gap = float(sorted_eigenvals[0] - sorted_eigenvals[1])
        else:
            spectral_gap = 0.0
        
        return {
            'largest_eigenvalue': largest_eigenval,
            'spectral_gap': spectral_gap
        }
    except:
        # Fallback for cases where eigendecomposition fails
        return {
            'largest_eigenvalue': estimate_largest_eigenvalue(adj_matrix),
            'spectral_gap': None
        }


def estimate_largest_eigenvalue(adj_matrix: np.ndarray, iterations: int = 100) -> float:
    """
    Estimate largest eigenvalue using power iteration method.
    """
    n = adj_matrix.shape[0]
    if n == 0:
        return 0.0
    
    # Initialize random vector
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    
    for _ in range(iterations):
        x_new = adj_matrix @ x
        x_new_norm = np.linalg.norm(x_new)
        
        if x_new_norm == 0:
            return 0.0
        
        x = x_new / x_new_norm
    
    # Estimate eigenvalue
    Ax = adj_matrix @ x
    eigenval = x.T @ Ax
    
    return float(abs(eigenval))


def validate_sparse_approximation(
    original_adj: Union[np.ndarray, csr_matrix],
    sparse_adj: Union[np.ndarray, csr_matrix],
    tolerance: float = 0.1
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that sparse approximation preserves essential graph properties.
    
    Args:
        original_adj: Original dense adjacency matrix
        sparse_adj: Sparse approximation
        tolerance: Tolerance for property preservation (fractional difference)
        
    Returns:
        Tuple of (is_valid, validation_report)
    """
    # Compute properties for both matrices
    orig_props = compute_graph_properties(original_adj)
    sparse_props = compute_graph_properties(sparse_adj)
    
    validation_report = {
        'original_properties': orig_props,
        'sparse_properties': sparse_props,
        'property_differences': {},
        'validation_passed': True,
        'failed_checks': []
    }
    
    # Compare key properties
    properties_to_check = [
        ('density', 'density'),
        ('mean_degree', 'mean_degree'),
        ('clustering_coefficient', 'clustering_coefficient'),
        ('num_connected_components', 'num_connected_components')
    ]
    
    for orig_key, sparse_key in properties_to_check:
        orig_val = orig_props[orig_key]
        sparse_val = sparse_props[sparse_key]
        
        if isinstance(orig_val, (int, float)) and orig_val != 0:
            diff = abs(orig_val - sparse_val) / abs(orig_val)
            validation_report['property_differences'][orig_key] = {
                'original': orig_val,
                'sparse': sparse_val,
                'relative_difference': diff,
                'tolerance': tolerance,
                'passed': diff <= tolerance
            }
            
            if diff > tolerance:
                validation_report['validation_passed'] = False
                validation_report['failed_checks'].append({
                    'property': orig_key,
                    'difference': diff,
                    'threshold': tolerance
                })
        elif orig_val == 0 and sparse_val == 0:
            validation_report['property_differences'][orig_key] = {
                'original': orig_val,
                'sparse': sparse_val,
                'relative_difference': 0.0,
                'tolerance': tolerance,
                'passed': True
            }
        else:
            # Special handling for zero values
            abs_diff = abs(orig_val - sparse_val)
            validation_report['property_differences'][orig_key] = {
                'original': orig_val,
                'sparse': sparse_val,
                'absolute_difference': abs_diff,
                'tolerance': tolerance,
                'passed': abs_diff <= tolerance
            }
            
            if abs_diff > tolerance:
                validation_report['validation_passed'] = False
                validation_report['failed_checks'].append({
                    'property': orig_key,
                    'difference': abs_diff,
                    'threshold': tolerance
                })
    
    # Additional checks
    # Edge preservation check
    if issparse(original_adj):
        orig_edges = original_adj.nnz
    else:
        orig_edges = np.count_nonzero(original_adj)
    
    if issparse(sparse_adj):
        sparse_edges = sparse_adj.nnz
    else:
        sparse_edges = np.count_nonzero(sparse_adj)
    
    edge_preservation = sparse_edges / orig_edges if orig_edges > 0 else 0.0
    validation_report['edge_preservation'] = {
        'original_edges': orig_edges,
        'sparse_edges': sparse_edges,
        'preservation_ratio': edge_preservation
    }
    
    # If too few edges are preserved, flag as invalid regardless of tolerance
    if edge_preservation < 0.1:  # Less than 10% of edges preserved
        validation_report['validation_passed'] = False
        validation_report['failed_checks'].append({
            'property': 'edge_preservation',
            'issue': f'Only {edge_preservation:.2%} of edges preserved',
            'threshold': 0.1
        })
    
    return validation_report['validation_passed'], validation_report


def compute_approximation_quality(
    original_adj: Union[np.ndarray, csr_matrix],
    sparse_adj: Union[np.ndarray, csr_matrix]
) -> Dict[str, float]:
    """
    Compute quantitative measures of approximation quality.
    
    Args:
        original_adj: Original adjacency matrix
        sparse_adj: Sparse approximation
        
    Returns:
        Dictionary of quality metrics
    """
    if issparse(original_adj):
        orig_dense = original_adj.toarray()
    else:
        orig_dense = original_adj
    
    if issparse(sparse_adj):
        sparse_dense = sparse_adj.toarray()
    else:
        sparse_dense = sparse_adj
    
    # Compute Frobenius norm of difference
    diff_matrix = orig_dense - sparse_dense
    frobenius_error = float(np.linalg.norm(diff_matrix, 'fro'))
    frobenius_original = float(np.linalg.norm(orig_dense, 'fro'))
    relative_frobenius_error = frobenius_error / frobenius_original if frobenius_original > 0 else 0.0
    
    # Compute element-wise accuracy
    total_elements = orig_dense.size
    matching_elements = np.sum(np.isclose(orig_dense, sparse_dense, rtol=1e-5, atol=1e-8))
    element_accuracy = matching_elements / total_elements
    
    # Compute rank similarity (for larger matrices, estimate)
    try:
        orig_rank = np.linalg.matrix_rank(orig_dense)
        sparse_rank = np.linalg.matrix_rank(sparse_dense)
        rank_similarity = min(orig_rank, sparse_rank) / max(orig_rank, sparse_rank) if max(orig_rank, sparse_rank) > 0 else 0.0
    except:
        # For large matrices, estimate rank using trace of squared matrix
        orig_trace_sq = np.trace(orig_dense @ orig_dense.T)
        sparse_trace_sq = np.trace(sparse_dense @ sparse_dense.T)
        rank_similarity = sparse_trace_sq / orig_trace_sq if orig_trace_sq > 0 else 0.0
    
    return {
        'frobenius_error': frobenius_error,
        'relative_frobenius_error': relative_frobenius_error,
        'element_accuracy': element_accuracy,
        'rank_similarity': rank_similarity
    }