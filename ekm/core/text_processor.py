"""
Text processing module for EKM - Handles text chunking and preprocessing
"""
import re
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
from ..providers.base import BaseLLM, BaseEmbeddings


class TextProcessor:
    """Handles text chunking and preprocessing for EKM."""
    
    def __init__(self, min_chunk_size: int = 512, max_chunk_size: int = 2048, semantic_threshold: float = 0.82):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.semantic_threshold = semantic_threshold

    async def chunk_text(self, text: str) -> List[str]:
        """Chunk text according to paper specifications."""
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) <= 2:
            return [text]

        # Get embeddings for sentences
        embeddings = await self._get_sentence_embeddings(sentences)
        chunks = []
        current_chunk = [sentences[0]]
        current_emb = np.array(embeddings[0])

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(current_emb, embeddings[i])
            if len(" ".join(current_chunk)) >= self.min_chunk_size and sim < self.semantic_threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
                current_emb = np.array(embeddings[i])
            else:
                current_chunk.append(sentences[i])
                n = len(current_chunk)
                current_emb = (current_emb * (n-1) + np.array(embeddings[i])) / n

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    async def _get_sentence_embeddings(self, sentences: List[str]) -> List[List[float]]:
        """Get embeddings for sentences."""
        # This would be injected from the main EKM class
        # For now, we'll make this abstract
        raise NotImplementedError("This method should be implemented with embedding provider injection")

    def _cosine_similarity(self, v1, v2):
        """Calculate cosine similarity between two vectors."""
        a = np.array(v1)
        b = np.array(v2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0: 
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticTextProcessor(TextProcessor):
    """Semantic text processor that uses embeddings for chunking."""
    
    def __init__(self, embeddings: BaseEmbeddings, min_chunk_size: int = 512, max_chunk_size: int = 2048, semantic_threshold: float = 0.82):
        super().__init__(min_chunk_size, max_chunk_size, semantic_threshold)
        self.embeddings = embeddings

    async def _get_sentence_embeddings(self, sentences: List[str]) -> List[List[float]]:
        """Get embeddings for sentences using the provided embedding provider."""
        return await self.embeddings.embed_documents(sentences)