"""
Unit tests for the attention module.
Tests cover: SparseAttention, MultiHeadAttention, retrieval heads, and adaptive weighting.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock


class TestSparseAttention:
    """Tests for SparseAttention class."""
    
    def test_sparse_attention_init(self):
        """Test SparseAttention initialization."""
        from ekm.core.attention import SparseAttention
        
        attn = SparseAttention(d_model=768, num_heads=8, sparsity_factor=0.1)
        
        assert attn.d_model == 768
        assert attn.num_heads == 8
        assert attn.sparsity_factor == 0.1
        assert attn.d_k == 768 // 8
    
    def test_sparse_attention_forward_basic(self, sample_embeddings_small):
        """Test SparseAttention forward pass with basic inputs."""
        from ekm.core.attention import SparseAttention
        
        d_model = 64
        attn = SparseAttention(d_model=d_model, num_heads=4, sparsity_factor=0.5)
        
        # Create Q, K, V with batch dimension
        batch_size = 1
        seq_len = 10
        Q = sample_embeddings_small[:seq_len].reshape(batch_size, seq_len, d_model)
        K = sample_embeddings_small[:seq_len].reshape(batch_size, seq_len, d_model)
        V = sample_embeddings_small[:seq_len].reshape(batch_size, seq_len, d_model)
        
        output, weights = attn.forward(Q, K, V)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights is not None
    
    def test_sparse_attention_with_graph_structure(self, sample_embeddings_small, sample_adjacency_matrix):
        """Test SparseAttention respects graph structure."""
        from ekm.core.attention import SparseAttention
        
        d_model = 64
        attn = SparseAttention(d_model=d_model, num_heads=4, sparsity_factor=0.5)
        
        batch_size = 1
        seq_len = 10
        Q = sample_embeddings_small[:seq_len].reshape(batch_size, seq_len, d_model)
        K = sample_embeddings_small[:seq_len].reshape(batch_size, seq_len, d_model)
        V = sample_embeddings_small[:seq_len].reshape(batch_size, seq_len, d_model)
        
        # Add graph structure
        graph_structure = sample_adjacency_matrix
        
        output, weights = attn.forward(Q, K, V, graph_structure=graph_structure)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_sparse_attention_softmax_numerical_stability(self):
        """Test softmax implementation handles large values."""
        from ekm.core.attention import SparseAttention
        
        attn = SparseAttention(d_model=64, num_heads=4)
        
        # Test with large values that could cause overflow
        x = np.array([[1000.0, 1001.0, 999.0]])
        result = attn.softmax(x)
        
        # Should sum to 1 and not contain NaN/Inf
        assert np.allclose(np.sum(result), 1.0)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention class."""
    
    def test_multihead_attention_init(self):
        """Test MultiHeadAttention initialization."""
        from ekm.core.attention import MultiHeadAttention
        
        attn = MultiHeadAttention(d_model=768, num_heads=8)
        
        assert attn.d_model == 768
        assert attn.num_heads == 8
        assert attn.d_k == 96
    
    def test_multihead_attention_init_invalid_heads(self):
        """Test MultiHeadAttention raises error for incompatible dimensions."""
        from ekm.core.attention import MultiHeadAttention
        
        # 768 is not divisible by 7
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=768, num_heads=7)
    
    def test_multihead_attention_forward(self, sample_embeddings_small):
        """Test MultiHeadAttention forward pass."""
        from ekm.core.attention import MultiHeadAttention
        
        d_model = 64
        attn = MultiHeadAttention(d_model=d_model, num_heads=4)
        
        batch_size = 1
        seq_len = 10
        Q = sample_embeddings_small[:seq_len].reshape(batch_size, seq_len, d_model)
        K = sample_embeddings_small[:seq_len].reshape(batch_size, seq_len, d_model)
        V = sample_embeddings_small[:seq_len].reshape(batch_size, seq_len, d_model)
        
        output, weights = attn.forward(Q, K, V)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, 4, seq_len, seq_len)  # (batch, num_heads, seq_q, seq_k)
    
    def test_multihead_attention_split_combine_heads(self):
        """Test split_heads and combine_heads are inverse operations."""
        from ekm.core.attention import MultiHeadAttention
        
        d_model = 64
        num_heads = 4
        attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        
        batch_size = 2
        seq_len = 10
        x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        
        split = attn.split_heads(x)
        combined = attn.combine_heads(split)
        
        assert np.allclose(x, combined)


class TestSemanticRetrievalHead:
    """Tests for SemanticRetrievalHead class."""
    
    def test_semantic_head_compute_attention(self, sample_embeddings_small):
        """Test semantic attention computation."""
        from ekm.core.attention import SemanticRetrievalHead
        
        head = SemanticRetrievalHead(embedding_dim=64)
        
        query_embedding = sample_embeddings_small[0]
        gku_embeddings = [sample_embeddings_small[i] for i in range(1, 5)]
        
        scores = head.compute_attention(query_embedding, gku_embeddings)
        
        assert len(scores) == 4
        assert isinstance(scores, np.ndarray)
    
    def test_semantic_head_empty_gkus(self):
        """Test semantic head handles empty GKU list."""
        from ekm.core.attention import SemanticRetrievalHead
        
        head = SemanticRetrievalHead(embedding_dim=64)
        query_embedding = np.random.randn(64).astype(np.float32)
        
        scores = head.compute_attention(query_embedding, [])
        
        assert scores.size == 0


class TestStructuralRetrievalHead:
    """Tests for StructuralRetrievalHead class."""
    
    def test_structural_head_compute_attention(self):
        """Test structural attention computation."""
        from ekm.core.attention import StructuralRetrievalHead
        
        head = StructuralRetrievalHead()
        
        query_pattern = {'graphlets': {'triangle': 1, 'wedge': 2}, 'degree_stats': {'mean_degree': 2.5}}
        gku_patterns = [
            {'graphlets': {'triangle': 1, 'wedge': 2}, 'degree_stats': {'mean_degree': 2.3}},
            {'graphlets': {'triangle': 5, 'wedge': 6}, 'degree_stats': {'mean_degree': 4.0}},
        ]
        
        scores = head.compute_attention(query_pattern, gku_patterns)
        
        assert len(scores) == 2
        assert isinstance(scores, np.ndarray)
    
    def test_structural_head_empty_patterns(self):
        """Test structural head handles empty pattern list."""
        from ekm.core.attention import StructuralRetrievalHead
        
        head = StructuralRetrievalHead()
        query_pattern = {'graphlets': [1, 2, 3]}
        
        scores = head.compute_attention(query_pattern, [])
        
        assert scores.size == 0


class TestTemporalRetrievalHead:
    """Tests for TemporalRetrievalHead class."""
    
    def test_temporal_head_compute_attention(self):
        """Test temporal attention computation."""
        from ekm.core.attention import TemporalRetrievalHead
        
        head = TemporalRetrievalHead()
        
        query_time_context = {'timestamp': 1000, 'recency_weight': 0.8}
        gku_times = [
            {'created_at': 900, 'last_accessed': 950},
            {'created_at': 500, 'last_accessed': 600},
        ]
        
        scores = head.compute_attention(query_time_context, gku_times)
        
        assert len(scores) == 2
        assert isinstance(scores, np.ndarray)
        # More recent should score higher
        assert scores[0] >= scores[1]
    
    def test_temporal_head_none_context(self):
        """Test temporal head handles None query context."""
        from ekm.core.attention import TemporalRetrievalHead
        
        head = TemporalRetrievalHead()
        
        gku_times = [{'created_at': 900}, {'created_at': 500}]
        scores = head.compute_attention(None, gku_times)
        
        # Should return uniform scores when no time context
        assert len(scores) == 2


class TestAdaptiveHeadWeighting:
    """Tests for AdaptiveHeadWeighting class."""
    
    def test_adaptive_weighting_init_default(self):
        """Test default initialization."""
        from ekm.core.attention import AdaptiveHeadWeighting
        
        weighting = AdaptiveHeadWeighting()
        weights = weighting.get_current_weights()
        
        assert 'semantic' in weights
        assert 'structural' in weights
        assert 'temporal' in weights
        assert np.isclose(sum(weights.values()), 1.0)
    
    def test_adaptive_weighting_init_custom(self):
        """Test custom initialization."""
        from ekm.core.attention import AdaptiveHeadWeighting
        
        custom_weights = {'semantic': 0.7, 'structural': 0.2, 'temporal': 0.1}
        weighting = AdaptiveHeadWeighting(initial_weights=custom_weights)
        weights = weighting.get_current_weights()
        
        assert weights['semantic'] == pytest.approx(0.7, rel=0.1)
    
    def test_adaptive_weighting_update(self, sample_embeddings_small):
        """Test weight update based on feedback."""
        from ekm.core.attention import AdaptiveHeadWeighting
        
        weighting = AdaptiveHeadWeighting()
        initial_weights = weighting.get_current_weights().copy()
        
        query_embedding = sample_embeddings_small[0]
        feedback = {'semantic': 1.0, 'structural': 0.5, 'temporal': 0.2}
        
        weighting.update_weights_based_on_query(
            query_embedding=query_embedding,
            query_pattern=None,
            query_time_context=None,
            performance_feedback=feedback
        )
        
        updated_weights = weighting.get_current_weights()
        
        # Weights should have changed
        assert updated_weights != initial_weights


class TestAttentionContainers:
    """Tests for attention result containers."""
    
    def test_attention_heads_dataclass(self):
        """Test AttentionHeads container."""
        from ekm.core.attention import AttentionHeads
        
        semantic = np.array([0.5, 0.3, 0.2])
        structural = np.array([0.4, 0.4, 0.2])
        temporal = np.array([0.6, 0.2, 0.2])
        
        heads = AttentionHeads(
            semantic=semantic,
            structural=structural,
            temporal=temporal
        )
        
        assert np.array_equal(heads.semantic, semantic)
        assert np.array_equal(heads.structural, structural)
        assert np.array_equal(heads.temporal, temporal)
    
    def test_attention_result_dataclass(self):
        """Test AttentionResult container."""
        from ekm.core.attention import AttentionResult
        
        scores = np.array([0.8, 0.6, 0.4])
        weights = {'semantic': np.array([0.5]), 'structural': np.array([0.3])}
        interpretations = {'method': 'qkv', 'top_k': 3}
        
        result = AttentionResult(
            scores=scores,
            weights=weights,
            interpretations=interpretations
        )
        
        assert np.array_equal(result.scores, scores)
        assert result.interpretations['method'] == 'qkv'
