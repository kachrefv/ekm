"""Fixtures for EKM tests."""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock


@pytest.fixture
def mock_storage():
    """Mock storage for testing."""
    storage = AsyncMock()
    storage.get_gkus_by_workspace = AsyncMock(return_value=[])
    storage.get_akus_with_embeddings = AsyncMock(return_value=[])
    storage.save_episode = AsyncMock(return_value="episode_123")
    storage.save_akus = AsyncMock(return_value=["aku_1", "aku_2"])
    storage.save_relationships = AsyncMock(return_value=None)
    return storage


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value='{"summary": "test summary", "facts": ["fact1", "fact2"]}')
    return llm


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing."""
    embeddings = AsyncMock()
    embeddings.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    embeddings.embed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    return embeddings


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "This is a sample text for testing the EKM system. It contains multiple sentences to test the chunking and extraction functionality."


@pytest.fixture
def event_loop():
    """Create a new event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()