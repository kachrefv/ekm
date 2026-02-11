"""
Retrieval module for EKM - Handles knowledge retrieval using various mechanisms
"""
import asyncio
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from ..state import FocusBuffer
from ..attention import QKVAttentionRetriever
from ..scalability import ScalableEKMBackend
from ..rl_feedback import RLFeedbackSystem
from ...storage.base import BaseStorage
from ...providers.base import BaseLLM, BaseEmbeddings
from ..query_analysis import QueryGraphExtractor

# Optional import for visualization - may not be available in all environments
try:
    from .attention_viz import AttentionDebugger
    ATTENTION_VIZ_AVAILABLE = True
except ImportError:
    ATTENTION_VIZ_AVAILABLE = False
    AttentionDebugger = None

logger = logging.getLogger(__name__)


class RetrievalService:
    """Handles knowledge retrieval using various mechanisms."""
    
    def __init__(
        self,
        storage: BaseStorage,
        llm: BaseLLM,
        embeddings: BaseEmbeddings,
        scalable_backend: ScalableEKMBackend,
        rl_feedback_system: RLFeedbackSystem,
        semantic_threshold: float = 0.82
    ):
        self.storage = storage
        self.llm = llm
        self.embeddings = embeddings
        self.scalable_backend = scalable_backend
        self.rl_feedback_system = rl_feedback_system
        self.semantic_threshold = semantic_threshold
        self.semantic_threshold = semantic_threshold
        self.debugger = AttentionDebugger() if ATTENTION_VIZ_AVAILABLE else None
        self.query_extractor = QueryGraphExtractor(llm)

    async def retrieve(
        self,
        workspace_id: str,
        query: str,
        top_k: int = 5,
        focus_buffer: Optional[FocusBuffer] = None,
        use_rl_feedback: bool = True,
        use_sparse_attention: bool = True,
        use_adaptive_weights: bool = True,
        use_hierarchical_attention: bool = True,
        layers: Optional[List[str]] = None,
        debug_mode: bool = False,
        mode: str = 'hybrid'  # 'hybrid', 'episodic', or 'causal'
    ) -> Dict[str, Any]:
        """Retrieve relevant knowledge from the mesh using unified QKV attention-based mechanism.
        
        Args:
            workspace_id: Workspace to retrieve from
            query: Query string
            top_k: Number of results to return
            focus_buffer: Optional focus buffer for context
            use_rl_feedback: Enable RL-based reranking
            use_sparse_attention: Use sparse attention for efficiency
            use_adaptive_weights: Use adaptive head weighting
            use_hierarchical_attention: Use hierarchical attention
            layers: Which layers to retrieve from (overridden by mode)
            debug_mode: If True, include detailed attention debug info
            mode: Retrieval mode:
                'episodic': Retrieve only from episodic memory
                'causal': Retrieve from AKU/GKU using advanced attention for high performance
                'hybrid': Combine all layers (default)
        """
        # Handle modes by setting layers and attention flags
        if mode == 'episodic':
            layers = ['episodic']
        elif mode == 'causal':
            layers = ['aku', 'gku']
            use_sparse_attention = False # Max performance/accuracy usually wants dense for causal
            use_adaptive_weights = True
            use_hierarchical_attention = True
        
        if layers is None:
            layers = ['episodic', 'aku', 'gku']  # Default to all layers

        # Unified retrieval approach that properly integrates all components
        all_results = []

        # 1. Retrieve from attention mechanism (AKUs with sparse pattern tensors)
        if 'aku' in layers or 'gku' in layers:
            qkv_results = await self._retrieve_with_qkv_attention(
                workspace_id=workspace_id,
                query=query,
                top_k=top_k,
                focus_buffer=focus_buffer,
                use_sparse_attention=use_sparse_attention,
                use_adaptive_weights=use_adaptive_weights,
                use_hierarchical_attention=use_hierarchical_attention
            )

            # Add attention results to all results
            all_results.extend(qkv_results['results'])

        # 2. Retrieve from episodic layer if requested
        if 'episodic' in layers:
            episodic_results = await self._retrieve_episodic_layer(
                workspace_id=workspace_id,
                query=query,
                top_k=top_k
            )

            # Add episodic results to all results
            all_results.extend(episodic_results['results'])

        # 3. Apply RL-based reranking if enabled and we have results
        if use_rl_feedback and all_results and len(all_results) > 1:
            # Prepare focus buffer state for RL system
            focus_buffer_state = None
            if focus_buffer:
                focus_buffer_state = {item_id: item.current_weight 
                                    for item_id, item in focus_buffer.items.items()}
            
            # Get RL-ranked results
            try:
                rl_prediction = await self.rl_feedback_system.get_ranked_responses(
                    query=query,
                    candidate_responses=all_results,
                    focus_buffer_state=focus_buffer_state
                )
                
                # Use RL-ranked results
                sorted_results = rl_prediction.ranked_responses
                
                # Add RL-specific metadata
                metadata_extension = {
                    "rl_reranking_applied": True,
                    "rl_confidence_scores": rl_prediction.confidence_scores,
                    "rl_exploration_factor": rl_prediction.exploration_factor,
                    "rl_model_performance": self.rl_feedback_system.get_model_performance_metrics()
                }
            except Exception as e:
                logger.warning(f"RL feedback system failed, falling back to original ranking: {e}")
                # Fallback to original sorting
                sorted_results = sorted(all_results, key=lambda x: x.get('score', x.get('semantic_similarity', 0)), reverse=True)
                metadata_extension = {"rl_reranking_applied": False, "rl_error": str(e)}
        else:
            # Original sorting without RL
            sorted_results = sorted(all_results, key=lambda x: x.get('score', x.get('semantic_similarity', 0)), reverse=True)
            metadata_extension = {"rl_reranking_applied": False}

        # 4. Take top-k results
        final_results = sorted_results[:top_k]

        # 5. Calculate layer distribution for the final results
        layer_distribution = {
            'attention_mesh': len([r for r in final_results if r.get('layer', '') == 'attention_mesh']),
            'episodic': len([r for r in final_results if r.get('layer', '') == 'episodic']),
            'aku': len([r for r in final_results if r.get('item_type', '') == 'aku']),
            'gku': len([r for r in final_results if r.get('item_type', '') == 'gku'])
        }

        # 6. Generate debug info if debug_mode is enabled
        debug_info = None
        if debug_mode:
            try:
                # Collect GKU names from results for debugging
                gku_names = [r.get('gku_name', r.get('id', 'unknown')) for r in final_results]
                attention_result = {
                    'head_contributions': layer_distribution,
                    'weights': {'semantic': np.array([r.get('score', 0) for r in final_results])},
                    'raw_scores': {'semantic': np.array([r.get('score', 0) for r in final_results])},
                    'normalized_scores': {'semantic': np.array([r.get('score', 0) for r in final_results])}
                }
                if self.debugger:
                    debug_info = self.debugger.create_debug_info(
                        query=query,
                        gku_names=gku_names,
                        attention_result=attention_result
                    )
                    logger.info(f"Debug mode: captured attention info for {len(gku_names)} results")
                else:
                    logger.warning("Debug mode enabled but AttentionDebugger not available")
            except Exception as e:
                logger.warning(f"Failed to generate debug info: {e}")

        # 7. Return unified results with proper metadata
        result = {
            "results": final_results,
            "metadata": {
                "layer_fusion_applied": True,
                "layers_retrieved": layers,
                "total_results": len(final_results),
                "layer_distribution": layer_distribution,
                "active_nodes": len([r for r in final_results if r.get('layer', '') == 'attention_mesh']),
                "retrieval_method": "unified_qkv_attention_with_sparse_tensors",
                **metadata_extension  # Include RL-specific metadata
            }
        }
        
        if debug_info:
            result["debug_info"] = debug_info
        
        return result

    async def _retrieve_episodic_layer(
        self,
        workspace_id: str,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Retrieve from the episodic memory layer."""
        # 1. Embed query
        query_emb = await self.embeddings.embed_query(query)

        # 2. Find similar episodes using scalable backend if available
        # For now, we'll continue using the existing method, but in the future
        # we could enhance SQLStorage to use vector databases like pgvector
        episodes = await self.storage.find_similar_episodes(workspace_id, query_emb, threshold=0.3, limit=top_k * 3)

        results = []
        for episode in episodes:
            results.append({
                "id": episode['id'],
                "content": episode['content'],
                "summary": episode['summary'],
                "similarity": episode['similarity'],
                "score": episode['similarity'],
                "layer": "episodic",
                "created_at": episode.get('created_at')
            })

        return {
            "results": sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k],
            "metadata": {"layer": "episodic", "result_count": len(results)}
        }

    async def _fuse_layer_results(self, layer_results: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Fuse results from different layers using attention mechanisms as described in the paper."""
        # Calculate query embedding for similarity comparisons
        query_emb = await self.embeddings.embed_query(query)

        # Prepare results from all layers
        all_results = []

        # Process Attention Mesh (Hierarchical GKU->AKU) results - this is the main QKV attention mechanism
        if 'attention_mesh' in layer_results and layer_results['attention_mesh']['results']:
            for result in layer_results['attention_mesh']['results']:
                content = result.get('content', '')
                if not content:
                    similarity = 0.0
                else:
                    content_emb = await self.embeddings.embed_query(content)
                    similarity = self._cosine_similarity(query_emb, content_emb)

                all_results.append({
                    'id': result['id'],
                    'content': content,
                    'layer': 'attention_mesh',
                    'original_score': result.get('score', 0),
                    'semantic_similarity': similarity,
                    'relevance_breakdown': result.get('relevance_breakdown', {}),
                    'parent_gku_id': result.get('parent_gku_id'),
                    'metadata': result
                })

        # Process AKU results
        if 'aku' in layer_results and layer_results['aku']['results']:
            for result in layer_results['aku']['results']:
                # Calculate semantic similarity with query
                content = result.get('content', '')
                if not content:
                    similarity = 0.0
                else:
                    content_emb = await self.embeddings.embed_query(content)
                    similarity = self._cosine_similarity(query_emb, content_emb)

                all_results.append({
                    'id': result['id'],
                    'content': content,
                    'layer': 'aku',
                    'original_score': result.get('score', 0),
                    'semantic_similarity': similarity,
                    'metadata': result
                })

        # Process GKU results
        if 'gku' in layer_results and layer_results['gku']['results']:
            for result in layer_results['gku']['results']:
                # Calculate semantic similarity with query
                content = result.get('content', '')
                if not content:
                    similarity = 0.0
                else:
                    content_emb = await self.embeddings.embed_query(content)
                    similarity = self._cosine_similarity(query_emb, content_emb)

                all_results.append({
                    'id': result['id'],
                    'content': content,
                    'layer': 'gku',
                    'original_score': result.get('score', 0),
                    'semantic_similarity': similarity,
                    'metadata': result
                })

        # Process episodic results
        if 'episodic' in layer_results and layer_results['episodic']['results']:
            for result in layer_results['episodic']['results']:
                # Calculate semantic similarity with query
                content = result.get('content', '')
                if not content:
                    similarity = 0.0
                else:
                    content_emb = await self.embeddings.embed_query(content)
                    similarity = self._cosine_similarity(query_emb, content_emb)

                all_results.append({
                    'id': result['id'],
                    'content': content,
                    'summary': result.get('summary'),
                    'layer': 'episodic',
                    'semantic_similarity': similarity,
                    'similarity': result.get('similarity', 0),
                    'created_at': result.get('created_at'),
                    'metadata': result
                })

        # Sort results by semantic similarity (or a combination of scores)
        sorted_results = sorted(all_results, key=lambda x: x['semantic_similarity'], reverse=True)

        # Limit to top results
        top_results = sorted_results[:len(sorted_results)]  # Return all for now, could be limited by top_k

        return {
            "results": top_results,
            "metadata": {
                "layer_fusion_applied": True,
                "layers_retrieved": list(layer_results.keys()),
                "total_results": len(top_results),
                "layer_distribution": {
                    layer: len([r for r in top_results if r['layer'] == layer])
                    for layer in layer_results.keys()
                }
            }
        }

    async def _retrieve_with_qkv_attention(
        self,
        workspace_id: str,
        query: str,
        top_k: int = 5,
        focus_buffer: Optional[FocusBuffer] = None,
        use_sparse_attention: bool = True,
        use_adaptive_weights: bool = True,
        use_hierarchical_attention: bool = False
    ) -> Dict[str, Any]:
        """Retrieve using QKV attention mechanism with sparse attention and adaptive weighting."""
        # 1. Embed query
        query_emb = await self.embeddings.embed_query(query)

        # 2. Initialize QKV Attention Retriever with adaptive weighting option
        retriever = QKVAttentionRetriever(
            embedding_dim=len(query_emb),
            use_adaptive_weights=use_adaptive_weights,
            use_hierarchical_attention=use_hierarchical_attention
        )

        # 3. Prepare query pattern
        # Extract structural pattern from the query using QueryGraphExtractor
        query_pattern = await self.query_extractor.extract_query_graph(query)
        logger.debug(f"Extracted query pattern: {query_pattern['graphlets']}")

        # 4. Prepare query time context (current time)
        from datetime import datetime
        query_time_context = {
            'timestamp': datetime.now()
        }

        # 5. Perform QKV attention retrieval with sparse attention option
        result = await retriever.retrieve(
            storage=self.storage,
            workspace_id=workspace_id,
            query_embedding=query_emb,
            query_pattern=query_pattern,
            query_time_context=query_time_context,
            top_k=top_k,
            use_sparse_attention=use_sparse_attention
        )

        # 6. Get detailed information about the retrieved GKUs
        gku_details = []
        top_gku_ids = result.interpretations.get('top_gku_ids', [])
        if not top_gku_ids:
             # Fallback for HierarchicalAttention which uses 'selected_gku_ids'
             top_gku_ids = result.interpretations.get('selected_gku_ids', [])
        
        top_scores = result.scores
        
        for i, gku_id in enumerate(top_gku_ids):
            # Get GKU details and associated AKUs
            gku_info = await self.storage.get_gku(gku_id)
            if gku_info:
                # Get AKUs in this GKU
                aku_list = await self.storage.get_akus_in_gku(gku_id)
                
                # Get the combined score from result.scores
                score = float(top_scores[i]) if i < len(top_scores) else 0.0
                
                gku_details.append({
                    'id': gku_info['id'],
                    'name': gku_info['name'],
                    'description': gku_info['description'],
                    'akus': aku_list,
                    'score': score
                })

        # 7. Prepare results
        # 7. Prepare results - Flatten to AKUs and rank by similarity
        results = []
        seen_aku_ids = set()
        
        # Collect all candidate AKUs from the top GKUs
        candidate_akus = []
        
        for detail in gku_details:
            for aku in detail['akus']:
                if aku['id'] not in seen_aku_ids:
                    # Enrich AKU with GKU context
                    aku_with_context = aku.copy()
                    aku_with_context['gku_context'] = {
                        'id': detail['id'],
                        'name': detail['name'],
                        'description': detail['description'],
                        'score': detail['score'] # GKU relevance score
                    }
                    candidate_akus.append(aku_with_context)
                    seen_aku_ids.add(aku['id'])
        
        # Rank candidate AKUs
        # We value:
        # 1. Semantic similarity to the query (specific relevance)
        # 2. The relevance of the GKU it belongs to (contextual relevance)
        
        for aku in candidate_akus:
            # Calculate semantic similarity if embedding exists
            similarity = 0.0
            if aku.get('embedding'):
                 # Ensure embedding is in list format for calculation
                emb = aku['embedding']
                if isinstance(emb, str):
                    import json
                    emb = json.loads(emb)
                similarity = self._cosine_similarity(query_emb, emb)
            
            # Combined score: 70% semantic match, 30% GKU context relevance
            # This ensures we get specific answers but prioritized by the "right" context
            gku_score = aku['gku_context']['score']
            final_score = (similarity * 0.7) + (gku_score * 0.3)
            
            results.append({
                "id": aku['id'],
                "content": aku['content'],
                "source_gku": aku['gku_context']['name'],
                "layer": "attention_mesh", # Keep layer name for compatibility
                "score": final_score,
                "semantic_similarity": similarity,
                "gku_score": gku_score,
                "metadata": {
                    "gku_id": aku['gku_context']['id'],
                    "gku_name": aku['gku_context']['name'],
                    "gku_description": aku['gku_context']['description'],
                    "aku_original_metadata": aku.get('aku_metadata', {}),
                    "created_at": aku.get('created_at')
                },
                "relevance_breakdown": {
                    "semantic": similarity,
                    "structural": result.interpretations.get('average_structural_contribution', 0),
                    "temporal": result.interpretations.get('average_temporal_contribution', 0)
                }
            })

        # Sort by final score
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        # Take top_k results
        results = results[:top_k]

        # 8. Generate explanation for retrieval decisions
        from ..explainability import generate_retrieval_explanation
        explanation = generate_retrieval_explanation(
            query=query,
            results=sorted(results, key=lambda x: x['score'], reverse=True),
            attention_weights=result.weights,
            interpretations=retriever.interpret_attention_weights(result)
        )

        # 9. Return results with attention interpretations and explanations
        return {
            "results": sorted(results, key=lambda x: x['score'], reverse=True),
            "metadata": {
                "active_nodes": len(result.interpretations.get('top_gku_ids', [])),
                "attention_interpretations": retriever.interpret_attention_weights(result),
                "retrieval_method": "qkv_attention",
                "explanation": explanation
            }
        }

    def _cosine_similarity(self, v1, v2):
        """Calculate cosine similarity between two vectors."""
        a = np.array(v1)
        b = np.array(v2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0: 
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))