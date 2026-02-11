"""
Factory module for EKM - Creates EKM instances with proper dependency injection
"""
from typing import Dict, Any, Optional
from .mesh import EKM
from ..storage.base import BaseStorage
from ..providers.base import BaseLLM, BaseEmbeddings


def create_ekm_instance(
    storage: BaseStorage,
    llm: BaseLLM,
    embeddings: BaseEmbeddings,
    config: Optional[Dict[str, Any]] = None
) -> EKM:
    """
    Factory function to create an EKM instance with proper dependency injection.
    
    Args:
        storage: Storage backend for the EKM
        llm: Language model provider
        embeddings: Embedding provider
        config: Optional configuration dictionary
    
    Returns:
        EKM instance with all dependencies properly injected
    """
    return EKM(
        storage=storage,
        llm=llm,
        embeddings=embeddings,
        config=config or {}
    )

    if config:
        from .scalability import check_production_readiness
        try:
            check_production_readiness(config)
        except RuntimeError as e:
            # We log it but maybe we shouldn't crash the factory unless strict?
            # The plan said "explicitly raise error".
            raise e


def create_ekm_with_memory_storage(
    llm: BaseLLM,
    embeddings: BaseEmbeddings,
    config: Optional[Dict[str, Any]] = None
) -> EKM:
    """
    Factory function to create an EKM instance with memory storage.
    
    Args:
        llm: Language model provider
        embeddings: Embedding provider
        config: Optional configuration dictionary
    
    Returns:
        EKM instance with memory storage and proper dependencies
    """
    from ..storage.factory import create_storage
    storage = create_storage("memory")
    
    return create_ekm_instance(
        storage=storage,
        llm=llm,
        embeddings=embeddings,
        config=config
    )


def create_ekm_with_sql_storage(
    session,
    llm: BaseLLM,
    embeddings: BaseEmbeddings,
    config: Optional[Dict[str, Any]] = None
) -> EKM:
    """
    Factory function to create an EKM instance with SQL storage.
    
    Args:
        session: Database session
        llm: Language model provider
        embeddings: Embedding provider
        config: Optional configuration dictionary
    
    Returns:
        EKM instance with SQL storage and proper dependencies
    """
    from ..storage.factory import create_storage
    storage = create_storage("sql", session=session)
    
    return create_ekm_instance(
        storage=storage,
        llm=llm,
        embeddings=embeddings,
        config=config
    )