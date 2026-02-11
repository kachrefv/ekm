from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass

class BaseEmbeddings(ABC):
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings."""
        pass
