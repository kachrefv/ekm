"""
Unit tests for the storage module.
Tests cover: MemoryStorage implementation and BaseStorage interface.
"""
import pytest
import numpy as np
import asyncio


class TestMemoryStorageInit:
    """Tests for MemoryStorage initialization."""
    
    def test_memory_storage_init(self):
        """Test MemoryStorage initializes empty stores."""
        from ekm.storage.memory import MemoryStorage
        
        storage = MemoryStorage()
        
        assert storage.workspaces == {}
        assert storage.episodes == {}
        assert storage.akus == {}
        assert storage.gkus == {}
        assert storage.relationships == {}


class TestMemoryStorageWorkspace:
    """Tests for workspace operations."""
    
    @pytest.mark.asyncio
    async def test_get_workspace_creates_if_missing(self, memory_storage):
        """Test get_workspace creates workspace if it doesn't exist."""
        workspace = await memory_storage.get_workspace("new-workspace")
        
        assert workspace is not None
        assert workspace['id'] == "new-workspace"
    
    @pytest.mark.asyncio
    async def test_get_workspace_returns_existing(self, memory_storage):
        """Test get_workspace returns existing workspace."""
        # Create workspace first
        await memory_storage.get_workspace("existing")
        
        # Get it again
        workspace = await memory_storage.get_workspace("existing")
        
        assert workspace['id'] == "existing"


class TestMemoryStorageEpisode:
    """Tests for episode operations."""
    
    @pytest.mark.asyncio
    async def test_save_episode(self, memory_storage):
        """Test saving an episode."""
        workspace_id = "test-workspace"
        embedding = [0.1] * 768
        
        episode_id = await memory_storage.save_episode(
            workspace_id=workspace_id,
            content="Test episode content",
            summary="Test summary",
            embedding=embedding,
            metadata={"title": "Test Episode"}
        )
        
        assert episode_id is not None
        assert episode_id in memory_storage.episodes
    
    @pytest.mark.asyncio
    async def test_find_similar_episodes(self, memory_storage):
        """Test finding similar episodes."""
        workspace_id = "test-workspace"
        
        # Save some episodes with known embeddings
        embedding1 = np.array([1.0] + [0.0] * 767).tolist()
        embedding2 = np.array([0.9, 0.1] + [0.0] * 766).tolist()
        embedding3 = np.array([0.0, 1.0] + [0.0] * 766).tolist()
        
        await memory_storage.save_episode(workspace_id, "Content 1", "Summary 1", embedding1, {})
        await memory_storage.save_episode(workspace_id, "Content 2", "Summary 2", embedding2, {})
        await memory_storage.save_episode(workspace_id, "Content 3", "Summary 3", embedding3, {})
        
        # Query with embedding similar to embedding1
        query_embedding = np.array([0.95, 0.05] + [0.0] * 766).tolist()
        
        results = await memory_storage.find_similar_episodes(
            workspace_id, query_embedding, threshold=0.5, limit=2
        )
        
        assert len(results) <= 2
        if len(results) > 0:
            assert results[0]['similarity'] >= 0.5


class TestMemoryStorageAKU:
    """Tests for AKU operations."""
    
    @pytest.mark.asyncio
    async def test_save_single_aku(self, memory_storage):
        """Test saving a single AKU."""
        workspace_id = "test-workspace"
        
        akus = [{
            'content': 'Test AKU content',
            'embedding': [0.1] * 768,
            'aku_metadata': {'source': 'test'}
        }]
        
        aku_ids = await memory_storage.save_akus(workspace_id, None, akus)
        
        assert len(aku_ids) == 1
        assert aku_ids[0] in memory_storage.akus
    
    @pytest.mark.asyncio
    async def test_save_multiple_akus(self, memory_storage):
        """Test saving multiple AKUs."""
        workspace_id = "test-workspace"
        
        akus = [
            {'content': f'AKU {i}', 'embedding': [0.1 * i] * 768, 'aku_metadata': {}}
            for i in range(5)
        ]
        
        aku_ids = await memory_storage.save_akus(workspace_id, None, akus)
        
        assert len(aku_ids) == 5
    
    @pytest.mark.asyncio
    async def test_find_similar_akus(self, memory_storage):
        """Test finding similar AKUs."""
        workspace_id = "test-workspace"
        
        # Save AKUs with known embeddings
        np.random.seed(42)
        akus = []
        for i in range(5):
            embedding = np.random.randn(768).astype(np.float32)
            embedding = (embedding / np.linalg.norm(embedding)).tolist()
            akus.append({
                'content': f'AKU content {i}',
                'embedding': embedding,
                'aku_metadata': {}
            })
        
        await memory_storage.save_akus(workspace_id, None, akus)
        
        # Query with first AKU's embedding
        query_embedding = akus[0]['embedding']
        
        results = await memory_storage.find_similar_akus(
            workspace_id, query_embedding, threshold=0.0, limit=3
        )
        
        assert len(results) <= 3
    
    @pytest.mark.asyncio
    async def test_get_akus_by_ids(self, memory_storage):
        """Test retrieving AKUs by IDs."""
        workspace_id = "test-workspace"
        
        akus = [
            {'content': 'AKU 1', 'embedding': [0.1] * 768, 'aku_metadata': {}},
            {'content': 'AKU 2', 'embedding': [0.2] * 768, 'aku_metadata': {}}
        ]
        
        aku_ids = await memory_storage.save_akus(workspace_id, None, akus)
        
        retrieved = await memory_storage.get_akus_by_ids(aku_ids)
        
        assert len(retrieved) == 2
        assert retrieved[0]['content'] in ['AKU 1', 'AKU 2']
    
    @pytest.mark.asyncio
    async def test_archive_akus(self, memory_storage):
        """Test archiving AKUs."""
        workspace_id = "test-workspace"
        
        akus = [{'content': 'AKU to archive', 'embedding': [0.1] * 768, 'aku_metadata': {}}]
        aku_ids = await memory_storage.save_akus(workspace_id, None, akus)
        
        await memory_storage.archive_akus(aku_ids)
        
        assert memory_storage.akus[aku_ids[0]]['is_archived'] is True
    
    @pytest.mark.asyncio
    async def test_get_akus_with_embeddings(self, memory_storage):
        """Test getting AKUs with embeddings."""
        workspace_id = "test-workspace"
        
        akus = [
            {'content': 'AKU with embedding', 'embedding': [0.1] * 768, 'aku_metadata': {}},
            {'content': 'AKU without embedding', 'embedding': None, 'aku_metadata': {}}
        ]
        
        await memory_storage.save_akus(workspace_id, None, akus)
        
        results = await memory_storage.get_akus_with_embeddings(workspace_id)
        
        # Should only return AKUs with embeddings
        for aku in results:
            assert aku.get('embedding') is not None


class TestMemoryStorageRelationships:
    """Tests for relationship operations."""
    
    @pytest.mark.asyncio
    async def test_save_relationships(self, memory_storage):
        """Test saving relationships."""
        workspace_id = "test-workspace"
        
        # First create some AKUs
        akus = [
            {'content': 'AKU 1', 'embedding': [0.1] * 768, 'aku_metadata': {}},
            {'content': 'AKU 2', 'embedding': [0.2] * 768, 'aku_metadata': {}}
        ]
        aku_ids = await memory_storage.save_akus(workspace_id, None, akus)
        
        # Create relationships
        relationships = [{
            'source_aku_id': aku_ids[0],
            'target_aku_id': aku_ids[1],
            'semantic_similarity': 0.8,
            'temporal_proximity': 0.9,
            'causal_weight': 0.5
        }]
        
        await memory_storage.save_relationships(workspace_id, relationships)
        
        assert len(memory_storage.relationships) == 1
    
    @pytest.mark.asyncio
    async def test_get_aku_relationships(self, memory_storage):
        """Test getting relationships for AKUs."""
        workspace_id = "test-workspace"
        
        # Create AKUs and relationships
        akus = [
            {'content': 'AKU 1', 'embedding': [0.1] * 768, 'aku_metadata': {}},
            {'content': 'AKU 2', 'embedding': [0.2] * 768, 'aku_metadata': {}},
            {'content': 'AKU 3', 'embedding': [0.3] * 768, 'aku_metadata': {}}
        ]
        aku_ids = await memory_storage.save_akus(workspace_id, None, akus)
        
        relationships = [
            {
                'source_aku_id': aku_ids[0],
                'target_aku_id': aku_ids[1],
                'semantic_similarity': 0.8
            },
            {
                'source_aku_id': aku_ids[1],
                'target_aku_id': aku_ids[2],
                'semantic_similarity': 0.7
            }
        ]
        await memory_storage.save_relationships(workspace_id, relationships)
        
        # Get relationships for first two AKUs
        results = await memory_storage.get_aku_relationships(workspace_id, aku_ids[:2])
        
        assert len(results) >= 1


class TestMemoryStorageGKU:
    """Tests for GKU operations."""
    
    @pytest.mark.asyncio
    async def test_save_gku(self, memory_storage):
        """Test saving a GKU."""
        workspace_id = "test-workspace"
        
        gku_id = await memory_storage.save_gku(
            workspace_id=workspace_id,
            name="Test GKU",
            description="A test global knowledge unit",
            centroid_embedding=[0.1] * 768,
            pattern_signature={'graphlets': [1, 2, 3]},
            cluster_metadata={'size': 5}
        )
        
        assert gku_id is not None
        assert gku_id in memory_storage.gkus
    
    @pytest.mark.asyncio
    async def test_get_gku(self, memory_storage):
        """Test retrieving a GKU."""
        workspace_id = "test-workspace"
        
        gku_id = await memory_storage.save_gku(
            workspace_id=workspace_id,
            name="Retrievable GKU",
            description="Test",
            centroid_embedding=[0.1] * 768
        )
        
        gku = await memory_storage.get_gku(gku_id)
        
        assert gku is not None
        assert gku['name'] == "Retrievable GKU"
    
    @pytest.mark.asyncio
    async def test_get_gku_not_found(self, memory_storage):
        """Test retrieving non-existent GKU."""
        gku = await memory_storage.get_gku("non-existent-id")
        
        assert gku is None
    
    @pytest.mark.asyncio
    async def test_get_gkus_by_workspace(self, memory_storage):
        """Test getting all GKUs in a workspace."""
        workspace_id = "test-workspace"
        
        # Save multiple GKUs
        for i in range(3):
            await memory_storage.save_gku(
                workspace_id=workspace_id,
                name=f"GKU {i}",
                description=f"Description {i}"
            )
        
        gkus = await memory_storage.get_gkus_by_workspace(workspace_id)
        
        assert len(gkus) == 3
    
    @pytest.mark.asyncio
    async def test_associate_akus_with_gku(self, memory_storage):
        """Test associating AKUs with a GKU."""
        workspace_id = "test-workspace"
        
        # Create AKUs
        akus = [
            {'content': f'AKU {i}', 'embedding': [0.1] * 768, 'aku_metadata': {}}
            for i in range(3)
        ]
        aku_ids = await memory_storage.save_akus(workspace_id, None, akus)
        
        # Create GKU
        gku_id = await memory_storage.save_gku(
            workspace_id=workspace_id,
            name="Parent GKU"
        )
        
        # Associate
        await memory_storage.associate_akus_with_gku(gku_id, aku_ids)
        
        # Verify
        associated_akus = await memory_storage.get_akus_in_gku(gku_id)
        
        assert len(associated_akus) == 3
    
    @pytest.mark.asyncio
    async def test_get_akus_in_gku(self, memory_storage):
        """Test getting AKUs in a GKU."""
        workspace_id = "test-workspace"
        
        # Create and associate AKUs with GKU
        akus = [
            {'content': 'Member AKU', 'embedding': [0.1] * 768, 'aku_metadata': {}}
        ]
        aku_ids = await memory_storage.save_akus(workspace_id, None, akus)
        
        gku_id = await memory_storage.save_gku(workspace_id, "GKU with members")
        await memory_storage.associate_akus_with_gku(gku_id, aku_ids)
        
        members = await memory_storage.get_akus_in_gku(gku_id)
        
        assert len(members) == 1
        assert members[0]['content'] == 'Member AKU'


class TestMemoryStorageCosineSimilarity:
    """Tests for cosine similarity helper."""
    
    def test_cosine_similarity_identical(self, memory_storage):
        """Test identical vectors have similarity 1."""
        vec = [1.0, 2.0, 3.0]
        similarity = memory_storage._cosine_similarity(vec, vec)
        
        assert np.isclose(similarity, 1.0)
    
    def test_cosine_similarity_orthogonal(self, memory_storage):
        """Test orthogonal vectors have similarity 0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = memory_storage._cosine_similarity(vec1, vec2)
        
        assert np.isclose(similarity, 0.0)
    
    def test_cosine_similarity_opposite(self, memory_storage):
        """Test opposite vectors have similarity -1."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = memory_storage._cosine_similarity(vec1, vec2)
        
        assert np.isclose(similarity, -1.0)


class TestSerializationHelpers:
    """Tests for serialization helpers."""
    
    def test_serialize_embedding_list(self, memory_storage):
        """Test serializing a list embedding."""
        embedding = [0.1, 0.2, 0.3]
        serialized = memory_storage._serialize_embedding(embedding)
        
        assert serialized == embedding
    
    def test_serialize_embedding_numpy(self, memory_storage):
        """Test serializing a numpy embedding."""
        embedding = np.array([0.1, 0.2, 0.3])
        serialized = memory_storage._serialize_embedding(embedding)
        
        assert isinstance(serialized, list)
        assert serialized == [0.1, 0.2, 0.3]
    
    def test_serialize_embedding_none(self, memory_storage):
        """Test serializing None embedding."""
        serialized = memory_storage._serialize_embedding(None)
        
        assert serialized is None
