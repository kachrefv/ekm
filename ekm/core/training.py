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
import logging

logger = logging.getLogger(__name__)


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

            # 6. Save AKUs, with the date each fact is about when the text states one
            fact_times = extraction_result.get('fact_times') or []
            akus_to_save = []
            for i, (content, emb) in enumerate(zip(aku_contents, aku_embeddings)):
                aku = {"content": content, "embedding": emb}
                event_time = fact_times[i] if i < len(fact_times) else None
                if event_time:
                    aku["metadata"] = {"event_time": event_time}
                akus_to_save.append(aku)

            aku_ids = await self.storage.save_akus(workspace_id, episode_id, akus_to_save)

            # 7. Add the AKUs just saved to this workspace's vector index.
            # (This used to call add_akus_batch, which embedded and saved every
            # AKU a second time, without its episode, and indexed only the copies.)
            await self.scalable_backend.index_akus(workspace_id, aku_ids, aku_embeddings)

            # 8. Generate relationships to earlier AKUs in the workspace.
            # AKUs from this chunk are linked by the local loop below instead.
            relationships = []
            this_chunk = {str(a) for a in aku_ids}

            for current_id, current_emb in zip(aku_ids, aku_embeddings):
                try:
                    neighbors = await self.scalable_backend.search_similar_akus(
                        workspace_id, current_emb, k=15 + len(this_chunk)
                    )
                except Exception as e:
                    logger.warning(f"Neighbour search failed for AKU {current_id}; "
                                   f"only same-chunk links will be created: {e}")
                    continue

                for neighbor_id, sim in neighbors:
                    if neighbor_id in this_chunk:
                        continue
                    # Cosine, the same measure and threshold as the local loop
                    if sim > 0.7:
                        relationships.append({
                            'source_aku_id': current_id,
                            'target_aku_id': neighbor_id,
                            'semantic_similarity': sim,
                            'temporal_proximity': 0.0 # Will be updated by graph manager if time info avail
                        })
            
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