"""Unit tests for attention mechanisms."""
import pytest
import numpy as np
from unittest.mock import AsyncMock, Mock
from ekm.core.attention import QKVAttentionRetriever, SparseAttention, MultiHeadAttention


@pytest.mark.asyncio
async def test_qkv_attention_retriever_initialization():
    """Test initialization of QKVAttentionRetriever."""
    retriever = QKVAttentionRetriever(embedding_dim=768, num_heads=8)
    
    assert retriever.semantic_head is not None
    assert retriever.structural_head is not None
    assert retriever.temporal_head is not None
    assert retriever.multihead_attn is not None
    assert retriever.sparse_attn is not None


@pytest.mark.asyncio
async def test_sparse_attention_initialization():
    """Test initialization of SparseAttention."""
    sparse_attn = SparseAttention(d_model=768, num_heads=8, sparsity_factor=0.1)
    
    assert sparse_attn.d_model == 768
    assert sparse_attn.num_heads == 8
    assert sparse_attn.sparsity_factor == 0.1
    assert sparse_attn.d_k == 768 // 8  # 96


@pytest.mark.asyncio
async def test_multihead_attention_initialization():
    """Test initialization of MultiHeadAttention."""
    multihead_attn = MultiHeadAttention(d_model=768, num_heads=8)
    
    assert multihead_attn.d_model == 768
    assert multihead_attn.num_heads == 8
    assert multihead_attn.d_k == 768 // 8  # 96


@pytest.mark.asyncio
async def test_semantic_attention_computation():
    """Test semantic attention computation."""
    from ekm.core.attention import SemanticRetrievalHead
    
    head = SemanticRetrievalHead(embedding_dim=768)
    
    # Create sample embeddings
    query_embedding = np.random.rand(768)
    gku_embeddings = [np.random.rand(768) for _ in range(5)]
    
    # Compute attention
    attention_weights = head.compute_attention(query_embedding, gku_embeddings)
    
    # Verify output
    assert len(attention_weights) == 5
    assert np.allclose(np.sum(attention_weights), 1.0)  # Should sum to 1 (softmax)
    assert np.all(attention_weights >= 0)  # All weights should be non-negative