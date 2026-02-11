"""Conftest for EKM tests."""
import pytest
import asyncio
from unittest.mock import AsyncMock


@pytest.fixture(scope="session")
def event_loop():
    """Create a new event loop for session scope."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def auto_mock_asyncio():
    """Automatically mock asyncio sleep to speed up tests."""
    original_sleep = asyncio.sleep
    
    async def mock_sleep(duration):
        pass  # Don't actually sleep in tests
    
    asyncio.sleep = mock_sleep
    yield
    asyncio.sleep = original_sleep