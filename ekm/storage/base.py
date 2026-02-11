from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import uuid

class BaseStorage(ABC):
    @abstractmethod
    async def get_workspace(self, workspace_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def save_episode(self, workspace_id: str, content: str, summary: str, embedding: List[float], metadata: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    async def save_akus(self, workspace_id: str, episode_id: Optional[str], akus: List[Dict[str, Any]]) -> List[str]:
        pass

    @abstractmethod
    async def save_relationships(self, workspace_id: str, relationships: List[Dict[str, Any]]):
        pass

    @abstractmethod
    async def find_similar_episodes(self, workspace_id: str, embedding: List[float], threshold: float, limit: int) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def find_similar_akus(self, workspace_id: str, embedding: List[float], threshold: float, limit: int) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_aku_relationships(self, workspace_id: str, aku_ids: List[str]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_akus_by_ids(self, aku_ids: List[str]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def archive_akus(self, aku_ids: List[str]):
        pass

    @abstractmethod
    async def save_gku(self, workspace_id: str, name: str, description: str = "", 
                      centroid_embedding: Optional[List[float]] = None, 
                      pattern_signature: Optional[Dict[str, Any]] = None,
                      cluster_metadata: Optional[Dict[str, Any]] = None) -> str:
        pass

    @abstractmethod
    async def get_gku(self, gku_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_gkus_by_workspace(self, workspace_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def associate_akus_with_gku(self, gku_id: str, aku_ids: List[str]):
        pass

    @abstractmethod
    async def get_akus_in_gku(self, gku_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def get_akus_with_embeddings(self, workspace_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def update_gku_pattern_signature(self, gku_id: str, pattern_signature: Dict[str, Any]):
        pass

    # RL State Persistence (for continuous learning)
    @abstractmethod
    async def save_rl_state(self, model_id: str, weights: List[float], metadata: Optional[Dict[str, Any]] = None):
        """Save RL model weights and state for persistence across restarts.
        
        Args:
            model_id: Identifier for the RL model (e.g., 'global' or user-specific)
            weights: Model weights as a list of floats
            metadata: Optional metadata (e.g., training stats, version)
        """
        pass

    @abstractmethod
    async def load_rl_state(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load RL model weights and state.
        
        Args:
            model_id: Identifier for the RL model
            
        Returns:
            Dict with 'weights' and 'metadata' if found, None otherwise
        """
        pass
