"""Main EKM package."""
# Import main classes for easy access
from .core.mesh import EKM
from .core.agent import EKMAgent
from .core.attention import QKVAttentionRetriever
from .core.retrieval import RetrievalService
from .core.training import TrainingService

__version__ = "0.1.9"
__author__ = "EKM Development Team"
__all__ = [
    "EKM",
    "EKMAgent", 
    "QKVAttentionRetriever",
    "RetrievalService",
    "TrainingService"
]