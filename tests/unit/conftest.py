"""
Shared pytest fixtures for EKM tests.
"""
import pytest
import numpy as np
import asyncio
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock


# ============================================================================
# Mock Providers
# ============================================================================

class MockLLM:
    """Mock LLM provider for testing."""
    
    def __init__(self, default_response: str = "Mocked LLM response"):
        self.default_response = default_response
        self.call_count = 0
        self.last_prompt = None
        self.last_message = None
    
    async def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        """Generate a mock response."""
        self.call_count += 1
        self.last_prompt = system_prompt
        self.last_message = user_message
        return self.default_response


class MockEmbeddings:
    """Mock embeddings provider for testing."""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.call_count = 0
        self._seed = 42
    
    async def embed_query(self, text: str) -> List[float]:
        """Generate a deterministic mock embedding for a query."""
        self.call_count += 1
        # Generate deterministic embedding based on text hash
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.dimension).astype(np.float32)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for multiple documents."""
        self.call_count += 1
        embeddings = []
        for text in texts:
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(self.dimension).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding.tolist())
        return embeddings


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_llm():
    """Provides a mock LLM instance."""
    return MockLLM()


@pytest.fixture
def mock_embeddings():
    """Provides a mock embeddings instance with 768 dimensions."""
    return MockEmbeddings(dimension=768)


@pytest.fixture
def mock_embeddings_small():
    """Provides a mock embeddings instance with smaller dimensions for faster tests."""
    return MockEmbeddings(dimension=64)


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(10, 768).astype(np.float32)


@pytest.fixture
def sample_embeddings_small():
    """Generate smaller sample embeddings for faster tests."""
    np.random.seed(42)
    return np.random.randn(10, 64).astype(np.float32)


@pytest.fixture
def sample_adjacency_matrix():
    """Generate a sample adjacency matrix for graph tests."""
    np.random.seed(42)
    n = 10
    # Create a sparse adjacency matrix
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        # Connect each node to 2-3 random neighbors
        num_neighbors = np.random.randint(2, 4)
        neighbors = np.random.choice([j for j in range(n) if j != i], num_neighbors, replace=False)
        for j in neighbors:
            adj[i, j] = np.random.rand()
    return adj


@pytest.fixture
def sample_akus():
    """Generate sample AKU data for testing."""
    np.random.seed(42)
    akus = []
    for i in range(5):
        embedding = np.random.randn(768).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        akus.append({
            'id': f'aku-{i}',
            'content': f'This is sample AKU content number {i}',
            'embedding': embedding.tolist(),
            'workspace_id': 'test-workspace',
            'is_archived': False
        })
    return akus


@pytest.fixture
def sample_gkus():
    """Generate sample GKU data for testing."""
    np.random.seed(42)
    gkus = []
    for i in range(3):
        embedding = np.random.randn(768).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        gkus.append({
            'id': f'gku-{i}',
            'name': f'Global Knowledge Unit {i}',
            'description': f'Description for GKU {i}',
            'centroid_embedding': embedding.tolist(),
            'pattern_signature': {'graphlets': [i, i+1, i+2]},
            'workspace_id': 'test-workspace'
        })
    return gkus


@pytest.fixture
def memory_storage():
    """Provides an in-memory storage instance."""
    from ekm.storage.memory import MemoryStorage
    return MemoryStorage()


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Utility Functions for Tests
# ============================================================================

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def generate_random_embedding(dim: int = 768, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random normalized embedding."""
    if seed is not None:
        np.random.seed(seed)
    embedding = np.random.randn(dim).astype(np.float32)
    return embedding / np.linalg.norm(embedding)
