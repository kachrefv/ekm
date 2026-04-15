#!/usr/bin/env python3
"""
Quick test to verify async SQLAlchemy migration works.
"""
import asyncio
import uuid
import numpy as np
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from ekm.storage.sql import SQLStorage
from ekm.core.models import Base

async def test_async_storage():
    """Test basic async storage operations."""
    # Create async engine with in-memory SQLite
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create async session
    SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as session:
        storage = SQLStorage(session)
        
        # Test workspace creation
        workspace_id = str(uuid.uuid4())
        workspace_name = "test_workspace"
        
        # Save workspace
        await storage.save_workspace(workspace_id, workspace_name, "test_user")
        
        # Get workspace
        workspace = await storage.get_workspace(workspace_id)
        assert workspace is not None
        assert workspace["id"] == workspace_id
        assert workspace["name"] == workspace_name
        
        # Test episode creation
        embedding = np.random.randn(3072).tolist()
        episode_id = await storage.save_episode(
            workspace_id=workspace_id,
            content="Test episode content",
            summary="Test summary",
            embedding=embedding,
            metadata={"test": True}
        )
        assert episode_id is not None
        
        # Test AKU creation
        aku_data = {
            "content": "Test AKU content",
            "embedding": np.random.randn(3072).tolist(),
            "metadata": {"type": "test"}
        }
        aku_ids = await storage.save_akus(workspace_id, episode_id, [aku_data])
        assert len(aku_ids) == 1
        
        # Test retrieving AKUs
        akus = await storage.get_akus_by_workspace(workspace_id, limit=10)
        assert len(akus) == 1
        assert akus[0]["content"] == "Test AKU content"
        
        # Test relationships
        relationship = {
            "source_id": aku_ids[0],
            "target_id": aku_ids[0],  # self-relationship for test
            "relationship_type": "similar",
            "strength": 0.8,
            "metadata": {}
        }
        await storage.save_relationships(workspace_id, [relationship])
        
        print("✓ All async storage operations completed successfully")
        
        # Clean up (optional)
        await session.commit()
    
    print("\n✅ Async migration test PASSED!")

if __name__ == "__main__":
    asyncio.run(test_async_storage())