"""
Mock LLM and Embeddings for testing purposes.
"""
import numpy as np
from typing import List
from ..providers.base import BaseLLM, BaseEmbeddings


class MockLLM(BaseLLM):
    async def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        """Mock LLM that returns a simple response."""
        return f"This is a mock response to: {user_message[:50]}..."


class MockEmbeddings(BaseEmbeddings):
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

    async def embed_query(self, text: str) -> List[float]:
        """Create a deterministic embedding for the text."""
        # Simple hash-based embedding to ensure consistent results
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hex_dig = hash_obj.hexdigest()
        
        # Convert hex to floats
        embedding = []
        for i in range(0, len(hex_dig), 2):
            val = int(hex_dig[i:i+2], 16) / 255.0  # Normalize to [0, 1]
            embedding.append(val)
        
        # Pad or truncate to embedding_dim
        if len(embedding) < self.embedding_dim:
            embedding.extend([0.0] * (self.embedding_dim - len(embedding)))
        else:
            embedding = embedding[:self.embedding_dim]
        
        return embedding

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            embedding = await self.embed_query(text)
            embeddings.append(embedding)
        return embeddings