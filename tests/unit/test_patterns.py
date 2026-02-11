"""
Unit tests for the patterns module.
Tests cover: graphlet counting, random walks, degree statistics, and pattern signatures.
"""
import pytest
import numpy as np


class TestGraphletCounting:
    """Tests for graphlet counting functions."""
    
    def test_count_graphlets_triangle(self):
        """Test counting triangles in a graph."""
        from ekm.core.patterns import count_graphlets
        
        # Create a triangle graph
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=np.float32)
        
        graphlets = count_graphlets(adj)
        
        assert 'triangle' in graphlets
        assert graphlets['triangle'] >= 1
    
    def test_count_graphlets_path(self):
        """Test counting graphlets in a path graph."""
        from ekm.core.patterns import count_graphlets
        
        # Create a path: 0-1-2
        adj = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=np.float32)
        
        graphlets = count_graphlets(adj)
        
        assert 'wedge' in graphlets
        assert graphlets['wedge'] >= 1
        assert graphlets.get('triangle', 0) == 0
    
    def test_count_graphlets_empty(self):
        """Test counting graphlets in empty graph."""
        from ekm.core.patterns import count_graphlets
        
        adj = np.zeros((5, 5), dtype=np.float32)
        
        graphlets = count_graphlets(adj)
        
        assert graphlets.get('triangle', 0) == 0
        assert graphlets.get('wedge', 0) == 0
    
    def test_count_graphlets_larger_graph(self, sample_adjacency_matrix):
        """Test with larger random graph."""
        from ekm.core.patterns import count_graphlets
        
        graphlets = count_graphlets(sample_adjacency_matrix)
        
        assert isinstance(graphlets, dict)
        assert 'triangle' in graphlets or 'wedge' in graphlets


class TestRandomWalkPatterns:
    """Tests for random walk pattern computation."""
    
    def test_random_walk_basic(self):
        """Test basic random walk pattern extraction."""
        from ekm.core.patterns import compute_random_walk_patterns
        
        # Create a connected graph
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.float32)
        
        patterns = compute_random_walk_patterns(adj, walk_length=3, num_walks=5)
        
        assert patterns is not None
        assert isinstance(patterns, dict)
    
    def test_random_walk_disconnected(self):
        """Test random walk on disconnected graph."""
        from ekm.core.patterns import compute_random_walk_patterns
        
        # Disconnected graph
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.float32)
        
        patterns = compute_random_walk_patterns(adj, walk_length=3, num_walks=5)
        
        assert patterns is not None
    
    def test_random_walk_single_node(self):
        """Test random walk on single node graph."""
        from ekm.core.patterns import compute_random_walk_patterns
        
        adj = np.array([[0]], dtype=np.float32)
        
        patterns = compute_random_walk_patterns(adj, walk_length=3, num_walks=2)
        
        assert patterns is not None


class TestDegreeStatistics:
    """Tests for degree statistics computation."""
    
    def test_degree_stats_basic(self):
        """Test basic degree statistics."""
        from ekm.core.patterns import compute_degree_statistics
        
        # Create a graph with known degrees
        adj = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        stats = compute_degree_statistics(adj)
        
        assert 'mean_degree' in stats
        assert 'max_degree' in stats
        assert 'min_degree' in stats
        assert 'std_degree' in stats
    
    def test_degree_stats_regular_graph(self):
        """Test degree stats on regular graph (all same degree)."""
        from ekm.core.patterns import compute_degree_statistics
        
        # Ring graph: each node has degree 2
        adj = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ], dtype=np.float32)
        
        stats = compute_degree_statistics(adj)
        
        assert np.isclose(stats['mean_degree'], 2.0)
        assert np.isclose(stats['std_degree'], 0.0, atol=0.01)
    
    def test_degree_stats_empty_graph(self):
        """Test degree stats on empty graph."""
        from ekm.core.patterns import compute_degree_statistics
        
        adj = np.zeros((5, 5), dtype=np.float32)
        
        stats = compute_degree_statistics(adj)
        
        assert stats['mean_degree'] == 0
        assert stats['max_degree'] == 0


class TestSemanticCentroids:
    """Tests for semantic centroid computation."""
    
    def test_compute_centroids_basic(self):
        """Test basic centroid computation."""
        from ekm.core.patterns import compute_semantic_centroids
        
        np.random.seed(42)
        embeddings = np.random.randn(10, 64).tolist()
        cluster_indices = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
        
        centroids = compute_semantic_centroids(embeddings, cluster_indices)
        
        assert len(centroids) == 3  # 3 clusters
        for centroid in centroids.values():
            assert len(centroid) == 64
    
    def test_compute_centroids_single_cluster(self):
        """Test centroid with single cluster."""
        from ekm.core.patterns import compute_semantic_centroids
        
        embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        cluster_indices = [0, 0, 0]
        
        centroids = compute_semantic_centroids(embeddings, cluster_indices)
        
        assert len(centroids) == 1
        assert 0 in centroids


class TestPatternSignatureExtraction:
    """Tests for comprehensive pattern signature extraction."""
    
    def test_extract_pattern_signature_basic(self, sample_adjacency_matrix):
        """Test basic pattern signature extraction."""
        from ekm.core.patterns import extract_pattern_signature
        
        signature = extract_pattern_signature(sample_adjacency_matrix)
        
        assert isinstance(signature, dict)
        assert 'graphlets' in signature or 'degree_stats' in signature
    
    def test_extract_pattern_signature_with_embeddings(self, sample_adjacency_matrix, sample_embeddings_small):
        """Test pattern signature with embeddings."""
        from ekm.core.patterns import extract_pattern_signature
        
        embeddings = sample_embeddings_small.tolist()
        cluster_labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
        
        signature = extract_pattern_signature(
            sample_adjacency_matrix,
            embeddings=embeddings,
            cluster_labels=cluster_labels
        )
        
        assert isinstance(signature, dict)
    
    def test_extract_pattern_signature_empty_graph(self):
        """Test pattern signature on empty graph."""
        from ekm.core.patterns import extract_pattern_signature
        
        adj = np.zeros((5, 5), dtype=np.float32)
        
        signature = extract_pattern_signature(adj)
        
        assert isinstance(signature, dict)


class TestPatternSimilarity:
    """Tests for pattern similarity calculation."""
    
    def test_pattern_similarity_identical(self):
        """Test similarity of identical patterns."""
        from ekm.core.patterns import calculate_pattern_similarity
        
        sig = {
            'graphlets': {'triangles': 5, 'wedges': 10},
            'degree_stats': {'mean': 2.5, 'std': 1.0}
        }
        
        similarity = calculate_pattern_similarity(sig, sig)
        
        assert np.isclose(similarity, 1.0)
    
    def test_pattern_similarity_different(self):
        """Test similarity of different patterns."""
        from ekm.core.patterns import calculate_pattern_similarity
        
        sig1 = {
            'graphlets': {'triangles': 5, 'wedges': 10},
            'degree_stats': {'mean': 2.5, 'std': 1.0}
        }
        sig2 = {
            'graphlets': {'triangles': 50, 'wedges': 100},
            'degree_stats': {'mean': 10.0, 'std': 5.0}
        }
        
        similarity = calculate_pattern_similarity(sig1, sig2)
        
        assert 0 <= similarity <= 1
        assert similarity < 1.0  # Should be different
    
    def test_pattern_similarity_custom_weights(self):
        """Test pattern similarity with custom weights."""
        from ekm.core.patterns import calculate_pattern_similarity
        
        sig1 = {
            'graphlets': {'triangles': 5},
            'degree_stats': {'mean': 2.5}
        }
        sig2 = {
            'graphlets': {'triangles': 10},
            'degree_stats': {'mean': 2.5}  # Same degree stats
        }
        
        # Weight degree stats higher
        weights = {'graphlets': 0.2, 'degree_stats': 0.8}
        
        similarity = calculate_pattern_similarity(sig1, sig2, weights=weights)
        
        assert 0 <= similarity <= 1
    
    def test_pattern_similarity_empty_signatures(self):
        """Test similarity of empty signatures."""
        from ekm.core.patterns import calculate_pattern_similarity
        
        sig1 = {}
        sig2 = {}
        
        similarity = calculate_pattern_similarity(sig1, sig2)
        
        # Should handle gracefully
        assert isinstance(similarity, (int, float))


class TestPatternStorageIntegration:
    """Tests for pattern storage integration."""
    
    @pytest.mark.asyncio
    async def test_extract_and_store_patterns(self, memory_storage, sample_adjacency_matrix):
        """Test extracting and storing patterns for a GKU."""
        from ekm.core.patterns import extract_and_store_patterns_for_gku
        
        workspace_id = "test-workspace"
        
        # Create a GKU first
        gku_id = await memory_storage.save_gku(
            workspace_id=workspace_id,
            name="Test GKU",
            description="Test"
        )
        
        # Extract and store patterns
        await extract_and_store_patterns_for_gku(
            storage=memory_storage,
            gku_id=gku_id,
            adjacency_matrix=sample_adjacency_matrix
        )
        
        # Verify GKU was updated
        gku = await memory_storage.get_gku(gku_id)
        
        assert gku is not None
        # Pattern signature should be stored
        assert gku.get('pattern_signature') is not None
