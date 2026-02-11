"""
Training module for EKM - Handles knowledge ingestion and training
"""
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from .text_processor import SemanticTextProcessor
from .knowledge.extraction import KnowledgeExtractor
from .scalability import ScalableEKMBackend
from ..storage.base import BaseStorage
from ..providers.base import BaseLLM, BaseEmbeddings
from .utils import cosine_similarity


class TrainingService:
    """Handles knowledge ingestion and training for EKM."""
    
    def __init__(
        self,
        storage: BaseStorage,
        llm: BaseLLM,
        embeddings: BaseEmbeddings,
        scalable_backend: ScalableEKMBackend,
        min_chunk_size: int = 512,
        max_chunk_size: int = 2048,
        semantic_threshold: float = 0.82
    ):
        self.storage = storage
        self.llm = llm
        self.embeddings = embeddings
        self.scalable_backend = scalable_backend
        self.text_processor = SemanticTextProcessor(
            embeddings=embeddings,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            semantic_threshold=semantic_threshold
        )
        self.knowledge_extractor = KnowledgeExtractor(llm=llm, embeddings=embeddings)

    async def train(self, workspace_id: str, text: str, title: Optional[str] = None):
        """Train the EKM with new text data using single-step extraction as described in the paper."""
        # 1. Chunk text according to paper specifications
        chunks = await self.text_processor.chunk_text(text)

        for chunk in chunks:
            # 2. Single-step extraction: Extract both summary and AKUs in one LLM call
            extraction_result = await self.knowledge_extractor.single_step_extraction(chunk)

            # 3. Process the extraction result
            summary = extraction_result.get('summary', chunk[:200] + "...")
            aku_contents = extraction_result.get('facts', [])

            if not aku_contents:
                continue  # Skip if no AKUs were extracted

            # 4. Embed Episode and AKUs
            ep_embedding = await self.embeddings.embed_query(chunk)
            aku_embeddings = await self.embeddings.embed_documents(aku_contents)

            # 5. Save Episode
            episode_id = await self.storage.save_episode(
                workspace_id, chunk, summary, ep_embedding, {"title": title}
            )

            # 6. Save AKUs
            akus_to_save = []
            for content, emb in zip(aku_contents, aku_embeddings):
                akus_to_save.append({"content": content, "embedding": emb})

            aku_ids = await self.storage.save_akus(workspace_id, episode_id, akus_to_save)

            # 7. Add AKUs to scalable vector index
            await self.scalable_backend.add_akus_batch(
                workspace_id, 
                [{"content": content, "id": id} for content, id in zip(aku_contents, aku_ids)]
            )

            # 8. Generate Relationships
            # Simplified: connect all AKUs in the same episode
            # 8. Generate Relationships using Global Vector Search
            relationships = []
            
            # Search for global connections for each new AKU
            for i, (current_id, current_emb) in enumerate(zip(aku_ids, aku_embeddings)):
                # Search locally and globally
                try:
                    distances, neighbor_ids = self.scalable_backend.vector_index.search(
                        np.array(current_emb), k=15 # Search slightly more to ensure good matches
                    )
                    
                    for dist, neighbor_id in zip(distances, neighbor_ids):
                        if neighbor_id is None or neighbor_id == current_id:
                            continue
                            
                        # Convert distance to similarity consistent with ScalableEKMBackend
                        sim = 1.0 / (1.0 + float(dist))
                        
                        if sim > 0.7:
                            relationships.append({
                                'source_aku_id': current_id,
                                'target_aku_id': neighbor_id,
                                'semantic_similarity': sim,
                                'temporal_proximity': 0.0 # Will be updated by graph manager if time info avail
                            })
                except Exception as e:
                    # Fallback to local loop if index search fails
                    pass
            
            # Ensure local temporal connections (fallback/reinforcement)
            for i in range(len(aku_ids)):
                for j in range(i + 1, len(aku_ids)):
                    sim = cosine_similarity(aku_embeddings[i], aku_embeddings[j])
                    if sim > 0.7:
                        relationships.append({
                            'source_aku_id': aku_ids[i],
                            'target_aku_id': aku_ids[j],
                            'semantic_similarity': sim,
                            'temporal_proximity': 1.0  # Same episode
                        })

            if relationships:
                await self.storage.save_relationships(workspace_id, relationships)