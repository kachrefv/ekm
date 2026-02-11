"""
Consolidation Framework for EKM - Implements the four-phase sleep-inspired consolidation process:
REPLAY, CONSOLIDATE, REORGANIZE, and PRUNE phases.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import uuid
from ..storage.base import BaseStorage
from ..core.models import AKU, GKU
from ..providers.base import BaseLLM, BaseEmbeddings
from . import prompts

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False  # Approximate nearest neighbor library

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    """Result of the consolidation process."""
    consolidated_akus: List[str]  # IDs of newly created consolidated AKUs
    archived_akus: List[str]      # IDs of archived original AKUs
    created_gkus: List[str]       # IDs of newly created GKUs from consolidation
    compression_ratio: float      # Ratio of reduction in AKU count
    semantic_preservation: float  # Measure of how much meaning was preserved


class SleepConsolidator:
    """
    Implements the sleep-inspired consolidation process for EKM.
    Based on the biological memory consolidation during sleep.
    """
    
    def __init__(
        self, 
        storage: BaseStorage, 
        llm: BaseLLM, 
        embeddings: BaseEmbeddings,
        replay_threshold: float = 0.85,
        consolidate_threshold: float = 0.90,
        min_cluster_support: int = 3
    ):
        self.storage = storage
        self.llm = llm
        self.embeddings = embeddings
        self.replay_threshold = replay_threshold
        self.consolidate_threshold = consolidate_threshold
        self.min_cluster_support = min_cluster_support
    
    async def run_consolidation(self, workspace_id: str) -> ConsolidationResult:
        """
        Run the complete consolidation process: REPLAY -> CONSOLIDATE -> REORGANIZE -> PRUNE.
        
        Args:
            workspace_id: ID of the workspace to consolidate
        
        Returns:
            ConsolidationResult with details of the consolidation
        """
        logger.info(f"Starting consolidation for workspace {workspace_id}")
        
        # Phase 1: REPLAY - Build similarity graph of recent AKUs
        logger.info("Phase 1: REPLAY - Building similarity graph")
        similarity_graph = await self._replay_phase(workspace_id)
        
        # Phase 2: CONSOLIDATE - Detect and merge similar clusters
        logger.info("Phase 2: CONSOLIDATE - Detecting and merging duplicates")
        consolidated_akus, archived_akus = await self._consolidate_phase(
            workspace_id, similarity_graph
        )
        
        # Phase 3: REORGANIZE - Update pattern tensors and GKU memberships
        logger.info("Phase 3: REORGANIZE - Updating pattern tensors")
        created_gkus = await self._reorganize_phase(workspace_id)
        
        # Phase 4: PRUNE - Archive original AKUs and clean up
        logger.info("Phase 4: PRUNE - Archiving original AKUs")
        await self._prune_phase(archived_akus)
        
        # Calculate metrics
        original_count = len(consolidated_akus) + len(archived_akus)
        new_count = len(consolidated_akus)
        compression_ratio = (original_count - new_count) / original_count if original_count > 0 else 0.0

        # Estimate semantic preservation based on consolidation process
        semantic_preservation = self._calculate_semantic_preservation(original_count, len(created_gkus), compression_ratio)
        
        result = ConsolidationResult(
            consolidated_akus=consolidated_akus,
            archived_akus=archived_akus,
            created_gkus=created_gkus,
            compression_ratio=compression_ratio,
            semantic_preservation=semantic_preservation
        )
        
        logger.info(f"Consolidation completed. Compression ratio: {compression_ratio:.2%}")
        return result
    
    async def _replay_phase(self, workspace_id: str) -> Dict[str, List[Tuple[str, float]]]:
        """
        Phase 1: REPLAY - Build similarity graph of recent AKUs.

        Args:
            workspace_id: ID of the workspace to process

        Returns:
            Similarity graph as adjacency list: {aku_id: [(neighbor_id, similarity), ...]}
        """
        # Get all non-archived AKUs in the workspace
        all_akus = await self.storage.get_akus_with_embeddings(workspace_id)

        # Filter to only AKUs with embeddings
        akus_with_embeddings = [aku for aku in all_akus if aku['embedding'] is not None]

        if len(akus_with_embeddings) < 2:
            logger.info("Not enough AKUs with embeddings for consolidation")
            return {}

        # Build similarity graph
        similarity_graph = defaultdict(list)

        # Use approximate nearest neighbor search if FAISS is available and we have enough data
        if FAISS_AVAILABLE and len(akus_with_embeddings) > 100:
            logger.info(f"Using FAISS for similarity computation with {len(akus_with_embeddings)} AKUs")
            
            # Prepare embeddings matrix
            embeddings_matrix = np.array([aku['embedding'] for aku in akus_with_embeddings]).astype('float32')
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings_matrix = embeddings_matrix / norms
            
            # Build FAISS index
            dimension = embeddings_matrix.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity after normalization
            index.add(embeddings_matrix)
            
            # Find approximate nearest neighbors
            # We'll search for more neighbors than needed to account for threshold filtering
            k = min(len(akus_with_embeddings), 100)  # Search for top 100 neighbors
            similarities, indices = index.search(embeddings_matrix, k)
            
            # Build similarity graph from results
            for i, (aku, neighbor_indices, neighbor_similarities) in enumerate(zip(akus_with_embeddings, indices, similarities)):
                for j, (neighbor_idx, sim) in enumerate(zip(neighbor_indices, neighbor_similarities)):
                    if i != neighbor_idx and sim >= self.replay_threshold:
                        similarity_graph[aku['id']].append((akus_with_embeddings[neighbor_idx]['id'], float(sim)))
        else:
            # Fall back to brute force for smaller datasets or when FAISS is not available
            logger.info(f"Using brute force similarity computation with {len(akus_with_embeddings)} AKUs")
            
            # Compute pairwise similarities
            for i, aku1 in enumerate(akus_with_embeddings):
                for j, aku2 in enumerate(akus_with_embeddings):
                    if i != j:  # Don't compare AKU with itself
                        # Calculate cosine similarity between embeddings
                        sim = self._cosine_similarity(aku1['embedding'], aku2['embedding'])

                        # Only add to graph if similarity exceeds threshold
                        if sim >= self.replay_threshold:
                            similarity_graph[aku1['id']].append((aku2['id'], sim))

        logger.info(f"Built similarity graph with {len(similarity_graph)} nodes")
        return dict(similarity_graph)
    
    async def _consolidate_phase(
        self, 
        workspace_id: str, 
        similarity_graph: Dict[str, List[Tuple[str, float]]]
    ) -> Tuple[List[str], List[str]]:
        """
        Phase 2: CONSOLIDATE - Detect and merge similar AKUs.
        
        Args:
            workspace_id: ID of the workspace to process
            similarity_graph: Similarity graph from replay phase
        
        Returns:
            Tuple of (new_aku_ids, archived_aku_ids)
        """
        # Find connected components in the similarity graph (potential clusters to merge)
        clusters = self._find_connected_components(similarity_graph)
        
        new_aku_ids = []
        archived_aku_ids = []
        
        for cluster in clusters:
            if len(cluster) < 2:
                # Skip clusters with only one AKU
                continue
            
            if len(cluster) < self.min_cluster_support:
                # Skip clusters that don't meet minimum support
                continue
            
            # Merge AKUs in this cluster
            merged_aku_id = await self._merge_akus_in_cluster(workspace_id, cluster)
            if merged_aku_id:
                new_aku_ids.append(merged_aku_id)
                # Archive original AKUs in the cluster
                archived_aku_ids.extend(cluster)
        
        logger.info(f"Consolidated {len(archived_aku_ids)} AKUs into {len(new_aku_ids)} new AKUs")
        return new_aku_ids, archived_aku_ids
    
    async def _reorganize_phase(self, workspace_id: str) -> List[str]:
        """
        Phase 3: REORGANIZE - Update pattern tensors and GKU memberships.
        
        Args:
            workspace_id: ID of the workspace to process
        
        Returns:
            List of newly created GKU IDs
        """
        # Re-cluster AKUs to form new GKUs based on updated embeddings
        # This would typically involve calling the clustering functionality
        from .knowledge.organization import form_gkus_from_akus
        
        # Form new GKUs from the remaining AKUs
        new_gku_ids = await form_gkus_from_akus(
            storage=self.storage,
            workspace_id=workspace_id,
            min_cluster_size=self.min_cluster_support
        )
        
        logger.info(f"Created {len(new_gku_ids)} new GKUs during reorganization")
        return new_gku_ids
    
    async def _prune_phase(self, aku_ids_to_archive: List[str]) -> None:
        """
        Phase 4: PRUNE - Archive original AKUs and clean up.
        
        Args:
            aku_ids_to_archive: List of AKU IDs to archive
        """
        # Mark AKUs as archived in batch
        if aku_ids_to_archive:
            await self.storage.archive_akus(aku_ids_to_archive)
        
        logger.info(f"Archived {len(aku_ids_to_archive)} AKUs")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)
        
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return float(dot_product / (norm_v1 * norm_v2))
    
    def _find_connected_components(self, graph: Dict[str, List[Tuple[str, float]]]) -> List[List[str]]:
        """
        Find connected components in the similarity graph using BFS.
        
        Args:
            graph: Similarity graph as adjacency list
        
        Returns:
            List of clusters, where each cluster is a list of AKU IDs
        """
        visited = set()
        components = []
        
        for node in graph:
            if node not in visited:
                # BFS to find all connected nodes
                component = []
                queue = [node]
                visited.add(node)
                
                while queue:
                    current = queue.pop(0)
                    component.append(current)
                    
                    # Add neighbors to queue if not visited
                    for neighbor, _ in graph.get(current, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                components.append(component)
        
        return components
    
    async def _merge_akus_in_cluster(self, workspace_id: str, cluster_aku_ids: List[str]) -> str:
        """
        Merge AKUs in a cluster into a single comprehensive AKU.
        
        Args:
            workspace_id: ID of the workspace
            cluster_aku_ids: List of AKU IDs to merge
        
        Returns:
            ID of the newly created merged AKU
        """
        # Get the AKUs to merge in batch
        aku_details = await self.storage.get_akus_by_ids(cluster_aku_ids)
        all_embeddings = [aku['embedding'] for aku in aku_details if aku.get('embedding') is not None]
        
        if not aku_details:
            raise ValueError("No valid AKUs to merge")
        
        # Create a comprehensive content by combining all AKU contents
        combined_content = self._combine_contents([aku['content'] for aku in aku_details])
        
        # Generate a new embedding for the merged AKU
        new_embedding = await self._generate_merged_embedding(all_embeddings)
        
        # Use LLM to create a comprehensive, conflict-resolved statement
        comprehensive_content = await self._create_comprehensive_content(
            [aku['content'] for aku in aku_details]
        )
        
        # Save the new merged AKU
        new_aku_data = [{
            'content': comprehensive_content,
            'embedding': new_embedding,
            'metadata': {
                'consolidated_from': cluster_aku_ids,
                'consolidation_timestamp': str(uuid.uuid4()), # Placeholder for actual timestamp
                'original_count': len(cluster_aku_ids)
            }
        }]
        
        new_aku_ids = await self.storage.save_akus(
            workspace_id=workspace_id,
            episode_id=None,  # Merged AKUs don't belong to a specific episode
            akus=new_aku_data
        )
        
        return new_aku_ids[0] if new_aku_ids else None
    
    def _combine_contents(self, contents: List[str]) -> str:
        """Combine multiple AKU contents into a single string."""
        return " ".join(contents)
    
    async def _generate_merged_embedding(self, embeddings: List[List[float]]) -> List[float]:
        """Generate a merged embedding by averaging the input embeddings."""
        if not embeddings:
            return []
        
        # Convert to numpy array and compute mean
        embeddings_array = np.array(embeddings)
        mean_embedding = np.mean(embeddings_array, axis=0)
        return mean_embedding.tolist()
    
    def _sanitize_input(self, content: str) -> str:
        """
        Sanitize input content to prevent prompt injection attacks.
        """
        # Remove potential prompt injection markers
        sanitized = content.replace("<|system|>", "").replace("<|user|>", "").replace("<|assistant|>", "")
        sanitized = sanitized.replace("[SYSTEM]", "").replace("[USER]", "").replace("[ASSISTANT]", "")
        sanitized = sanitized.replace("System:", "").replace("User:", "").replace("Assistant:", "")
        
        # Limit length to prevent extremely long inputs
        max_length = 10000  # Adjust as needed
        if len(sanitized) > max_length:
            logger.warning(f"Truncating content from {len(sanitized)} to {max_length} characters")
            sanitized = sanitized[:max_length]
        
        return sanitized

    async def _create_comprehensive_content(self, contents: List[str]) -> str:
        """
        Use LLM to create a comprehensive, conflict-resolved content from multiple AKUs.
        """
        if len(contents) == 1:
            return contents[0]

        try:
            # Sanitize all input contents
            sanitized_contents = [self._sanitize_input(content) for content in contents]
            
            # Generate the merged content using the LLM
            response = await self.llm.generate(
                system_prompt=prompts.CONSOLIDATION_COMPREHENSIVE_SYSTEM_PROMPT,
                user_message=prompts.CONSOLIDATION_COMPREHENSIVE_PROMPT.format(
                    statements_list=chr(10).join([f"- {stmt}" for stmt in sanitized_contents])
                )
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating comprehensive content: {e}")
            # Fallback: concatenate contents
            return " ".join(contents)
    
    def _calculate_semantic_preservation(self, original_count: int, gku_count: int, compression_ratio: float) -> float:
        """
        Calculate an estimate of semantic preservation based on consolidation outcomes.
        
        Args:
            original_count: Number of original AKUs before consolidation
            gku_count: Number of GKUs created during reorganization
            compression_ratio: The calculated compression ratio from consolidation
            
        Returns:
            Estimated semantic preservation score between 0 and 1
        """
        if original_count == 0:
            return 1.0
            
        # A balanced approach: consider both organization (GKUs) and compression effects
        gku_ratio = gku_count / original_count if original_count > 0 else 0
        
        # Base preservation on how well we've organized knowledge (GKUs) without over-compressing
        # Too much compression might lose detail, too little might not consolidate effectively
        optimal_compression = 0.3  # Target compression ratio for best preservation
        compression_factor = 1 - abs(compression_ratio - optimal_compression) / max(optimal_compression, 0.7)
        compression_factor = max(0.5, min(1.0, compression_factor))  # Clamp between 0.5 and 1.0
        
        # Organization factor: having appropriate number of GKUs indicates good organization
        # If we have too few or too many GKUs relative to original size, preservation may suffer
        if original_count > 0:
            # Ideal: 1 GKU per 10-50 AKUs (adjustable based on knowledge domain)
            ideal_gku_range = (original_count / 50, original_count / 10)
            if ideal_gku_range[0] <= gku_count <= ideal_gku_range[1]:
                organization_factor = 1.0
            else:
                # Penalize extremes: too few or too many GKUs
                distance_from_ideal = min(
                    abs(gku_count - ideal_gku_range[0]),
                    abs(gku_count - ideal_gku_range[1])
                )
                organization_factor = max(0.5, 1.0 - (distance_from_ideal / max(gku_count, 1) * 0.5))
        else:
            organization_factor = 1.0
        
        # Combine factors for final preservation score
        preservation_score = 0.6 * organization_factor + 0.4 * compression_factor
        
        return preservation_score

    async def _get_aku_by_id(self, aku_id: str) -> Optional[Dict[str, Any]]:
        """Get AKU by ID from storage."""
        akus = await self.storage.get_akus_by_ids([aku_id])
        return akus[0] if akus else None

    async def _archive_aku(self, aku_id: str) -> None:
        """Mark an AKU as archived."""
        # This would update the AKU's is_archived flag in the database
        # Implementation would depend on the specific storage interface
        pass