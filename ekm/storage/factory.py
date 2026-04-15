"""
Storage Factory for EKM - Creates different storage implementations
"""

from ..storage.sql import SQLStorage
from ..storage.memory import MemoryStorage
from ..storage.base import BaseStorage
from sqlalchemy.ext.asyncio import AsyncSession


def create_storage(storage_type: str = "memory", **kwargs) -> BaseStorage:
    """
    Factory function to create storage instances.

    Args:
        storage_type: Type of storage to create ("memory" or "sql")
        **kwargs: Additional arguments for specific storage types

    Returns:
        Instance of BaseStorage
    """
    if storage_type.lower() == "memory":
        return MemoryStorage()
    elif storage_type.lower() == "sql":
        db_session = kwargs.get("db_session")
        if not db_session:
            raise ValueError("db_session is required for SQL storage")
        if not isinstance(db_session, AsyncSession):
            raise TypeError("db_session must be an AsyncSession instance for async operation")
        return SQLStorage(db=db_session)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")


def create_ekm_with_memory_storage(llm, embeddings, config=None):
    """
    Convenience function to create EKM with in-memory storage.
    
    Args:
        llm: LLM instance
        embeddings: Embeddings instance  
        config: Configuration dictionary
        
    Returns:
        EKM instance with memory storage
    """
    from ..core.mesh import EKM
    storage = MemoryStorage()
    return EKM(storage=storage, llm=llm, embeddings=embeddings, config=config)


def create_ekm_with_sql_storage(db_session, llm, embeddings, config=None):
    """
    Convenience function to create EKM with SQL storage.

    Args:
        db_session: SQLAlchemy AsyncSession
        llm: LLM instance
        embeddings: Embeddings instance
        config: Configuration dictionary

    Returns:
        EKM instance with SQL storage
    """
    from ..core.mesh import EKM
    storage = SQLStorage(db=db_session)
    return EKM(storage=storage, llm=llm, embeddings=embeddings, config=config)