"""
Advanced Embedding Module for EKM System
Implements multi-modal embeddings with domain-specific fine-tuning
"""
import numpy as np
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)


class MultiModalEmbedder:
    """
    Advanced embedding system that combines multiple embedding modalities:
    - Semantic embeddings (using sentence transformers)
    - Structural embeddings (based on document structure)
    - Contextual embeddings (domain-specific fine-tuned models)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", domain_model_name: Optional[str] = None):
        """
        Initialize the multi-modal embedder.
        
        Args:
            model_name: Base sentence transformer model
            domain_model_name: Optional domain-specific fine-tuned model
        """
        self.base_model = SentenceTransformer(model_name)
        self.domain_model = None
        
        if domain_model_name:
            try:
                self.domain_model = SentenceTransformer(domain_model_name)
                logger.info(f"Loaded domain-specific model: {domain_model_name}")
            except Exception as e:
                logger.warning(f"Could not load domain-specific model {domain_model_name}: {e}")
                logger.info("Falling back to base model only")
        
        # Get embedding dimensions
        dummy_text = ["dummy text for dimension check"]
        base_embedding = self.base_model.encode(dummy_text)[0]
        self.base_embedding_dim = len(base_embedding)
        
        if self.domain_model:
            domain_embedding = self.domain_model.encode(dummy_text)[0]
            self.domain_embedding_dim = len(domain_embedding)
        else:
            self.domain_embedding_dim = self.base_embedding_dim
            
        # The combined embedding dimension is the same as base embedding
        # since we're doing weighted combination of same-sized vectors
        self.embedding_dim = self.base_embedding_dim
    
    def embed_texts(self, texts: List[str], weights: Optional[Dict[str, float]] = None) -> List[List[float]]:
        """
        Generate multi-modal embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            weights: Optional weights for different embedding modalities
            
        Returns:
            List of embedding vectors
        """
        if weights is None:
            weights = {
                'semantic': 0.5,
                'domain': 0.3,
                'structural': 0.2
            }
        
        # Get base semantic embeddings
        semantic_embeddings = self.base_model.encode(texts, convert_to_numpy=True)
        
        # Get domain-specific embeddings if available
        if self.domain_model:
            domain_embeddings = self.domain_model.encode(texts, convert_to_numpy=True)
        else:
            domain_embeddings = np.zeros_like(semantic_embeddings)
        
        # Calculate structural embeddings based on text properties
        structural_embeddings = self._calculate_structural_embeddings(texts)
        
        # Combine embeddings with weights
        combined_embeddings = []
        for i in range(len(texts)):
            combined = (
                weights['semantic'] * semantic_embeddings[i] +
                weights['domain'] * domain_embeddings[i] +
                weights['structural'] * structural_embeddings[i]
            )
            # Normalize the combined embedding
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
            combined_embeddings.append(combined.tolist())
        
        return combined_embeddings
    
    def _calculate_structural_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Calculate structural embeddings based on text properties.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Array of structural embeddings
        """
        # Create a base vector filled with structural features
        structural_embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        for i, text in enumerate(texts):
            # Calculate structural features
            text_len_norm = min(len(text) / 1000.0, 1.0)  # Normalized length
            word_count_norm = min(len(text.split()) / 100.0, 1.0)  # Normalized word count
            
            # Sentence structure features
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            sent_count_norm = min(len(sentences) / 10.0, 1.0)  # Normalized sentence count
            avg_sentence_len = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0
            avg_sent_len_norm = min(avg_sentence_len / 20.0, 1.0)  # Normalized avg sentence length
            
            # Complexity features
            complexity_score = self._calculate_complexity_score(text)
            
            # Distribute structural features across embedding dimensions
            # This spreads the features across the embedding space
            structural_embeddings[i, 0] = text_len_norm
            structural_embeddings[i, 1] = word_count_norm
            structural_embeddings[i, 2] = sent_count_norm
            structural_embeddings[i, 3] = avg_sent_len_norm
            structural_embeddings[i, 4] = complexity_score
            
            # Spread remaining dimensions with repeated patterns
            # This ensures the structural embedding has the right dimensionality
            for j in range(5, self.embedding_dim):
                # Use a pattern based on the structural features to populate remaining dims
                structural_embeddings[i, j] = (
                    (text_len_norm * (j % 3)) + 
                    (word_count_norm * ((j + 1) % 3)) + 
                    (sent_count_norm * ((j + 2) % 3))
                ) / 3.0
        
        # Normalize the structural embeddings to have similar scale to semantic embeddings
        norms = np.linalg.norm(structural_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        structural_embeddings = structural_embeddings / norms
        
        return structural_embeddings
    
    def _calculate_complexity_score(self, text: str) -> float:
        """
        Calculate a simple complexity score based on text characteristics.
        """
        # Count complex words (longer than 6 characters)
        words = text.split()
        complex_words = sum(1 for word in words if len(word) > 6)
        complexity = complex_words / len(words) if words else 0.0
        return min(complexity, 1.0)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query using multi-modal approach.
        """
        return self.embed_texts([query])[0]
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using multi-modal approach.
        """
        return self.embed_texts(documents)


class EnhancedEmbeddingProvider:
    """
    Wrapper class to integrate advanced embeddings with EKM system.
    """
    
    def __init__(self, embedder: MultiModalEmbedder):
        self.embedder = embedder
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a query."""
        return self.embedder.embed_query(text)
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        return self.embedder.embed_texts(texts)


# Legacy compatibility wrapper
class DomainSpecificEmbedder(MultiModalEmbedder):
    """
    Backwards-compatible wrapper for domain-specific embeddings.
    """
    pass