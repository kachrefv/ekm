"""
Unit tests for the clustering module.
Tests cover: spectral clustering, hierarchical clustering, ANN clustering, and GKU formation.
"""
import pytest
import numpy as np


class TestApproximateNearestNeighborClustering:
    """Tests for approximate nearest neighbor clustering."""
    
    def test_ann_clustering_basic(self):
        """Test basic ANN clustering."""
        from ekm.core.clustering import apply_approximate_nearest_neighbor_clustering
        
        np.random.seed(42)
        # Create 3 distinct clusters
        cluster1 = np.random.randn(10, 64) + np.array([5, 0] + [0]*62)
        cluster2 = np.random.randn(10, 64) + np.array([-5, 0] + [0]*62)
        cluster3 = np.random.randn(10, 64) + np.array([0, 5] + [0]*62)
        embeddings = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)
        
        labels = apply_approximate_nearest_neighbor_clustering(embeddings, n_clusters=3)
        
        assert labels is not None
        assert len(labels) == 30
        assert len(set(labels)) <= 3
    
    def test_ann_clustering_single_cluster(self):
        """Test ANN clustering with single cluster."""
        from ekm.core.clustering import apply_approximate_nearest_neighbor_clustering
        
        np.random.seed(42)
        embeddings = np.random.randn(10, 64).astype(np.float32)
        
        labels = apply_approximate_nearest_neighbor_clustering(embeddings, n_clusters=1)
        
        assert labels is not None
        assert len(set(labels)) == 1


class TestHierarchicalClustering:
    """Tests for hierarchical clustering."""
    
    def test_hierarchical_clustering_with_n_clusters(self):
        """Test hierarchical clustering with specified number of clusters."""
        from ekm.core.clustering import apply_hierarchical_clustering
        
        np.random.seed(42)
        embeddings = np.random.randn(20, 64).astype(np.float32)
        
        labels = apply_hierarchical_clustering(embeddings, n_clusters=3)
        
        assert labels is not None
        assert len(labels) == 20
        assert len(set(labels)) <= 3
    
    def test_hierarchical_clustering_with_distance_threshold(self):
        """Test hierarchical clustering with distance threshold."""
        from ekm.core.clustering import apply_hierarchical_clustering
        
        np.random.seed(42)
        embeddings = np.random.randn(20, 64).astype(np.float32)
        
        labels = apply_hierarchical_clustering(embeddings, distance_threshold=0.5)
        
        assert labels is not None
        assert len(labels) == 20


class TestMultiSimilarityClustering:
    """Tests for multi-similarity clustering."""
    
    def test_multi_similarity_clustering(self):
        """Test clustering with multiple similarity measures."""
        from ekm.core.knowledge.organization import apply_multi_similarity_clustering
        
        np.random.seed(42)
        embeddings = np.random.randn(20, 64).astype(np.float32)
        
        labels = apply_multi_similarity_clustering(embeddings, n_clusters=3)
        
        assert labels is not None
        assert len(labels) == 20


class TestSpectralClustering:
    """Tests for spectral clustering."""
    
    def test_spectral_clustering_basic(self):
        """Test basic spectral clustering."""
        from ekm.core.clustering import apply_spectral_clustering
        
        np.random.seed(42)
        embeddings = np.random.randn(20, 64).astype(np.float32)
        
        labels = apply_spectral_clustering(embeddings, n_clusters=3)
        
        assert labels is not None
        assert len(labels) == 20
        assert len(set(labels)) <= 3
    
    def test_spectral_clustering_auto_clusters(self):
        """Test spectral clustering with automatic cluster count."""
        from ekm.core.clustering import apply_spectral_clustering
        
        np.random.seed(42)
        embeddings = np.random.randn(20, 64).astype(np.float32)
        
        labels = apply_spectral_clustering(embeddings, n_clusters=None)
        
        assert labels is not None
        assert len(labels) == 20


class TestAdaptiveClustering:
    """Tests for adaptive clustering."""
    
    def test_adaptive_clustering_auto_params(self):
        """Test adaptive clustering with automatic parameter tuning."""
        from ekm.core.clustering import apply_adaptive_clustering
        
        np.random.seed(42)
        embeddings = np.random.randn(30, 64).astype(np.float32)
        
        labels, metrics = apply_adaptive_clustering(embeddings)
        
        assert labels is not None
        assert isinstance(metrics, dict)
        assert len(labels) == 30
    
    def test_adaptive_clustering_with_target(self):
        """Test adaptive clustering with target cluster count."""
        from ekm.core.clustering import apply_adaptive_clustering
        
        np.random.seed(42)
        embeddings = np.random.randn(30, 64).astype(np.float32)
        
        labels, metrics = apply_adaptive_clustering(embeddings, target_n_clusters=4)
        
        assert labels is not None
        assert len(set(labels)) <= 4


class TestIncrementalClusterUpdate:
    """Tests for incremental cluster updates."""
    
    def test_incremental_update(self):
        """Test incremental cluster update with new embeddings."""
        from ekm.core.knowledge.organization import incremental_cluster_update
        
        np.random.seed(42)
        current_embeddings = np.random.randn(20, 64).astype(np.float32)
        current_labels = np.array([0]*10 + [1]*10)
        new_embeddings = np.random.randn(5, 64).astype(np.float32)
        
        current_updated_labels, new_item_labels = incremental_cluster_update(
            current_embeddings, 
            current_labels, 
            new_embeddings
        )
        
        assert new_item_labels is not None
        assert len(new_item_labels) == 5


class TestClusterCentroids:
    """Tests for centroid computation."""
    
    def test_attention_weighted_centroids(self):
        """Test attention-weighted centroid computation."""
        from ekm.core.knowledge.organization import compute_attention_weighted_centroids
        
        np.random.seed(42)
        embeddings = np.random.randn(20, 64).astype(np.float32)
        labels = np.array([0]*10 + [1]*10)
        
        centroids, uncertainties = compute_attention_weighted_centroids(embeddings, labels)
        
        assert len(centroids) == 2  # 2 clusters
        assert len(uncertainties) == 2
        assert all(c.shape[0] == 64 for c in centroids)


class TestPatternSignatureExtraction:
    """Tests for pattern signature extraction."""
    
    def test_enhanced_pattern_signature(self):
        """Test enhanced pattern signature extraction."""
        from ekm.core.knowledge.organization import extract_enhanced_pattern_signature
        
        np.random.seed(42)
        embeddings = np.random.randn(20, 64).astype(np.float32)
        labels = np.array([0]*10 + [1]*10)
        
        signature = extract_enhanced_pattern_signature(embeddings, labels, cluster_idx=0)
        
        assert signature is not None
        assert isinstance(signature, dict)
        assert 'graphlets' in signature


class TestClusterQualityMetrics:
    """Tests for cluster quality metrics."""
    
    def test_compute_quality_metrics(self):
        """Test quality metrics computation."""
        from ekm.core.knowledge.organization import compute_cluster_quality_metrics
        
        np.random.seed(42)
        embeddings = np.random.randn(20, 64).astype(np.float32)
        labels = np.array([0]*10 + [1]*10)
        
        metrics = compute_cluster_quality_metrics(embeddings, labels)
        
        assert 'silhouette_score' in metrics or 'inertia' in metrics
        assert isinstance(metrics, dict)


class TestCosineSimilarity:
    """Tests for cosine similarity utility."""
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        from ekm.core.knowledge.organization import cosine_similarity_single
        
        vec = np.array([1.0, 2.0, 3.0])
        
        similarity = cosine_similarity_single(vec, vec)
        
        assert np.isclose(similarity, 1.0)
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        from ekm.core.knowledge.organization import cosine_similarity_single
        
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        
        similarity = cosine_similarity_single(vec1, vec2)
        
        assert np.isclose(similarity, 0.0)
    
    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        from ekm.core.knowledge.organization import cosine_similarity_single
        
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        
        similarity = cosine_similarity_single(vec1, vec2)
        
        assert np.isclose(similarity, -1.0)


class TestClusterSimilarity:
    """Tests for within-cluster similarity."""
    
    def test_avg_similarity_within_cluster(self):
        """Test average similarity calculation within cluster."""
        from ekm.core.knowledge.organization import calculate_avg_similarity_within_cluster
        
        np.random.seed(42)
        # Create tight cluster
        cluster_embeddings = np.random.randn(5, 64).astype(np.float32) * 0.1
        cluster_embeddings += np.ones((5, 64))  # Shift to make similar
        
        # Normalize
        cluster_embeddings = cluster_embeddings / np.linalg.norm(
            cluster_embeddings, axis=1, keepdims=True
        )
        
        avg_sim = calculate_avg_similarity_within_cluster(cluster_embeddings)
        
        assert avg_sim > 0.5  # Should be high for similar vectors
