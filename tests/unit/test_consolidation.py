"""
Unit tests for the consolidation module.
Tests cover: SleepConsolidator phases (REPLAY, CONSOLIDATE, REORGANIZE, PRUNE).
"""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio


class TestConsolidationResult:
    """Tests for ConsolidationResult dataclass."""
    
    def test_consolidation_result_creation(self):
        """Test ConsolidationResult initialization."""
        from ekm.core.consolidation import ConsolidationResult
        
        result = ConsolidationResult(
            consolidated_akus=['aku-1', 'aku-2'],
            archived_akus=['aku-3', 'aku-4'],
            created_gkus=['gku-1'],
            compression_ratio=0.25,
            semantic_preservation=0.95
        )
        
        assert len(result.consolidated_akus) == 2
        assert len(result.archived_akus) == 2
        assert len(result.created_gkus) == 1
        assert result.compression_ratio == 0.25
        assert result.semantic_preservation == 0.95


class TestSleepConsolidatorInit:
    """Tests for SleepConsolidator initialization."""
    
    def test_consolidator_default_init(self, memory_storage, mock_llm, mock_embeddings):
        """Test default initialization."""
        from ekm.core.consolidation import SleepConsolidator
        
        consolidator = SleepConsolidator(
            storage=memory_storage,
            llm=mock_llm,
            embeddings=mock_embeddings
        )
        
        assert consolidator.replay_threshold == 0.85
        assert consolidator.consolidate_threshold == 0.90
        assert consolidator.min_cluster_support == 3
    
    def test_consolidator_custom_thresholds(self, memory_storage, mock_llm, mock_embeddings):
        """Test custom threshold initialization."""
        from ekm.core.consolidation import SleepConsolidator
        
        consolidator = SleepConsolidator(
            storage=memory_storage,
            llm=mock_llm,
            embeddings=mock_embeddings,
            replay_threshold=0.7,
            consolidate_threshold=0.8,
            min_cluster_support=2
        )
        
        assert consolidator.replay_threshold == 0.7
        assert consolidator.consolidate_threshold == 0.8
        assert consolidator.min_cluster_support == 2


class TestCosineSimilarity:
    """Tests for cosine similarity helper method."""
    
    def test_cosine_similarity_identical(self, memory_storage, mock_llm, mock_embeddings):
        """Test cosine similarity with identical vectors."""
        from ekm.core.consolidation import SleepConsolidator
        
        consolidator = SleepConsolidator(memory_storage, mock_llm, mock_embeddings)
        
        vec = [1.0, 2.0, 3.0]
        similarity = consolidator._cosine_similarity(vec, vec)
        
        assert np.isclose(similarity, 1.0)
    
    def test_cosine_similarity_orthogonal(self, memory_storage, mock_llm, mock_embeddings):
        """Test cosine similarity with orthogonal vectors."""
        from ekm.core.consolidation import SleepConsolidator
        
        consolidator = SleepConsolidator(memory_storage, mock_llm, mock_embeddings)
        
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = consolidator._cosine_similarity(vec1, vec2)
        
        assert np.isclose(similarity, 0.0)
    
    def test_cosine_similarity_zero_vector(self, memory_storage, mock_llm, mock_embeddings):
        """Test cosine similarity with zero vector."""
        from ekm.core.consolidation import SleepConsolidator
        
        consolidator = SleepConsolidator(memory_storage, mock_llm, mock_embeddings)
        
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [0.0, 0.0, 0.0]
        similarity = consolidator._cosine_similarity(vec1, vec2)
        
        assert similarity == 0.0


class TestFindConnectedComponents:
    """Tests for connected component finding."""
    
    def test_find_single_component(self, memory_storage, mock_llm, mock_embeddings):
        """Test finding a single connected component."""
        from ekm.core.consolidation import SleepConsolidator
        
        consolidator = SleepConsolidator(memory_storage, mock_llm, mock_embeddings)
        
        # Create a fully connected graph
        graph = {
            'a': [('b', 0.9), ('c', 0.8)],
            'b': [('a', 0.9), ('c', 0.7)],
            'c': [('a', 0.8), ('b', 0.7)]
        }
        
        components = consolidator._find_connected_components(graph)
        
        assert len(components) == 1
        assert set(components[0]) == {'a', 'b', 'c'}
    
    def test_find_multiple_components(self, memory_storage, mock_llm, mock_embeddings):
        """Test finding multiple disconnected components."""
        from ekm.core.consolidation import SleepConsolidator
        
        consolidator = SleepConsolidator(memory_storage, mock_llm, mock_embeddings)
        
        # Create two disconnected components
        graph = {
            'a': [('b', 0.9)],
            'b': [('a', 0.9)],
            'c': [('d', 0.8)],
            'd': [('c', 0.8)]
        }
        
        components = consolidator._find_connected_components(graph)
        
        assert len(components) == 2
    
    def test_find_empty_graph(self, memory_storage, mock_llm, mock_embeddings):
        """Test with empty graph."""
        from ekm.core.consolidation import SleepConsolidator
        
        consolidator = SleepConsolidator(memory_storage, mock_llm, mock_embeddings)
        
        components = consolidator._find_connected_components({})
        
        assert components == []


class TestCombineContents:
    """Tests for combining AKU contents."""
    
    def test_combine_contents_basic(self, memory_storage, mock_llm, mock_embeddings):
        """Test basic content combination."""
        from ekm.core.consolidation import SleepConsolidator
        
        consolidator = SleepConsolidator(memory_storage, mock_llm, mock_embeddings)
        
        contents = ["First fact.", "Second fact.", "Third fact."]
        combined = consolidator._combine_contents(contents)
        
        assert "First fact" in combined
        assert "Second fact" in combined
        assert "Third fact" in combined
    
    def test_combine_contents_empty(self, memory_storage, mock_llm, mock_embeddings):
        """Test combining empty list."""
        from ekm.core.consolidation import SleepConsolidator
        
        consolidator = SleepConsolidator(memory_storage, mock_llm, mock_embeddings)
        
        combined = consolidator._combine_contents([])
        
        assert combined == ""


class TestGenerateMergedEmbedding:
    """Tests for merged embedding generation."""
    
    @pytest.mark.asyncio
    async def test_generate_merged_embedding_average(self, memory_storage, mock_llm, mock_embeddings):
        """Test merged embedding is average of inputs."""
        from ekm.core.consolidation import SleepConsolidator
        
        consolidator = SleepConsolidator(memory_storage, mock_llm, mock_embeddings)
        
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        
        merged = await consolidator._generate_merged_embedding(embeddings)
        
        expected = [1/3, 1/3, 1/3]
        assert np.allclose(merged, expected)
    
    @pytest.mark.asyncio
    async def test_generate_merged_embedding_single(self, memory_storage, mock_llm, mock_embeddings):
        """Test with single embedding."""
        from ekm.core.consolidation import SleepConsolidator
        
        consolidator = SleepConsolidator(memory_storage, mock_llm, mock_embeddings)
        
        embeddings = [[1.0, 2.0, 3.0]]
        merged = await consolidator._generate_merged_embedding(embeddings)
        
        assert merged == [1.0, 2.0, 3.0]


class TestConsolidationPhases:
    """Integration tests for consolidation phases."""
    
    @pytest.mark.asyncio
    async def test_replay_phase_builds_graph(self, memory_storage, mock_llm, mock_embeddings):
        """Test replay phase builds similarity graph."""
        from ekm.core.consolidation import SleepConsolidator
        
        # Setup: Add some AKUs to storage
        np.random.seed(42)
        workspace_id = "test-workspace"
        
        # Create test AKUs with similar embeddings
        akus = []
        base_embedding = np.random.randn(768).astype(np.float32)
        base_embedding = (base_embedding / np.linalg.norm(base_embedding)).tolist()
        
        for i in range(5):
            noise = np.random.randn(768).astype(np.float32) * 0.1
            embedding = np.array(base_embedding) + noise
            embedding = (embedding / np.linalg.norm(embedding)).tolist()
            akus.append({
                'content': f'Test AKU content {i}',
                'embedding': embedding,
                'aku_metadata': {}
            })
        
        await memory_storage.save_akus(workspace_id, None, akus)
        
        consolidator = SleepConsolidator(
            memory_storage, mock_llm, mock_embeddings,
            replay_threshold=0.5  # Lower threshold for testing
        )
        
        graph = await consolidator._replay_phase(workspace_id)
        
        assert isinstance(graph, dict)
    
    @pytest.mark.asyncio
    async def test_full_consolidation_empty_workspace(self, memory_storage, mock_llm, mock_embeddings):
        """Test consolidation on empty workspace."""
        from ekm.core.consolidation import SleepConsolidator
        
        consolidator = SleepConsolidator(memory_storage, mock_llm, mock_embeddings)
        
        result = await consolidator.run_consolidation("empty-workspace")
        
        assert result.consolidated_akus == []
        assert result.archived_akus == []
        assert result.compression_ratio == 0.0
