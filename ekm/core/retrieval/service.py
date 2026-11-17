"""
Retrieval module for EKM - Handles knowledge retrieval using various mechanisms

Every result carries a 'score' on one scale: cosine similarity between the
query and the item's embedding. Facts that belong to a well-matching GKU get a
small additive bonus on top (never a penalty). Episodes and facts are therefore
directly comparable when the layers are merged.
"""
import asyncio
import inspect
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

from ..state import FocusBuffer
from ..attention import QKVAttentionRetriever
from ..attention.mechanisms import DEFAULT_HEAD_WEIGHTS
from ..scalability import ScalableEKMBackend
from ..rl_feedback import RLFeedbackSystem
from ...storage.base import BaseStorage
from ...providers.base import BaseLLM, BaseEmbeddings
from ..query_analysis import QueryGraphExtractor

# Optional: AttentionDebugger needs only numpy, its plotting helpers need more.
try:
    from ..attention_viz import AttentionDebugger
    ATTENTION_VIZ_AVAILABLE = True
except ImportError:
    ATTENTION_VIZ_AVAILABLE = False
    AttentionDebugger = None

logger = logging.getLogger(__name__)

RETRIEVAL_MODES = ('hybrid', 'episodic', 'facts')
_MODE_ALIASES = {'causal': 'facts'}

# A fact inside a selected GKU gets up to this much added to its cosine score,
# in proportion to the GKU's relevance. It can reorder close calls but never
# lowers a fact below its own similarity (the old 70/30 blend usually did).
GKU_CONTEXT_BOOST = 0.1

# Episodes below this cosine similarity are not returned. Cosine on every
# backend now, so the value means the same thing everywhere.
EPISODIC_MIN_SIMILARITY = 0.3

# At most this many results from the same source episode (the episode itself
# plus facts extracted from it) before results from other sources get a turn.
MAX_RESULTS_PER_SOURCE = 2


def normalize_retrieval_mode(mode: Optional[str]) -> str:
    """
    One of RETRIEVAL_MODES. 'causal' (the old name for 'facts') is accepted
    with a deprecation warning; unknown modes fall back to 'hybrid', which is
    what they silently did before, but now with a warning.
    """
    key = (mode or 'hybrid').strip().lower()
    if key in _MODE_ALIASES:
        import warnings
        new = _MODE_ALIASES[key]
        warnings.warn(f"Retrieval mode '{key}' is deprecated; use '{new}'. It never followed "
                      f"cause-and-effect links, it retrieves facts.", DeprecationWarning, stacklevel=3)
        return new
    if key not in RETRIEVAL_MODES:
        logger.warning(f"Unknown retrieval mode '{mode}', using 'hybrid'")
        return 'hybrid'
    return key


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
        """
        Args:
            scalable_backend: kept for API compatibility; retrieval searches
                through `storage`, which is the source of truth for archived
                and newly added facts.
            semantic_threshold: kept for API compatibility. It is the chunking
                threshold, not a retrieval cut-off (see EPISODIC_MIN_SIMILARITY).
        """
        self.storage = storage
        self.llm = llm
        self.embeddings = embeddings
        self.scalable_backend = scalable_backend
        self.rl_feedback_system = rl_feedback_system
        self.semantic_threshold = semantic_threshold
        # One weighting for the life of the service, so feedback accumulates
        # (retrievers are created per query).
        from ..attention.mechanisms import AdaptiveHeadWeighting
        self.adaptive_weighting = AdaptiveHeadWeighting()
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
        mode: str = 'hybrid'  # 'hybrid', 'episodic', or 'facts' ('causal' = old name for 'facts')
    ) -> Dict[str, Any]:
        """Retrieve relevant knowledge from episodes and facts.

        Args:
            workspace_id: Workspace to retrieve from
            query: Query string
            top_k: Number of results to return
            focus_buffer: Facts the conversation is already about; GKUs that
                contain them get a bounded boost
            use_rl_feedback: Enable RL-based reranking (keeps the retrieval
                order until the RL model has learned from feedback)
            use_sparse_attention: Ignored; kept for API compatibility
            use_adaptive_weights: Read head weights from the shared adaptive
                weighting (changed only through record_head_feedback)
            use_hierarchical_attention: Select at least 5 GKUs instead of top_k
            layers: Which layers to retrieve from (overridden by mode)
            debug_mode: If True, include per-head attention debug info and an
                explanation of the ranking
            mode: Retrieval mode:
                'episodic': Retrieve only from episodic memory (raw source chunks)
                'facts': Retrieve only atomic facts (AKUs), via their GKUs plus
                    direct search. Nothing here follows cause-and-effect links:
                    relationships carry a causal_weight field, but nothing sets it.
                'causal': Deprecated alias for 'facts', kept so existing callers work
                'hybrid': Combine all layers (default)
        """
        mode = normalize_retrieval_mode(mode)

        if mode == 'episodic':
            layers = ['episodic']
        elif mode == 'facts':
            layers = ['aku', 'gku']
        if layers is None:
            layers = ['episodic', 'aku', 'gku']

        want_facts = 'aku' in layers or 'gku' in layers
        want_episodes = 'episodic' in layers

        # The query-graph extraction is an LLM call: start it first so it runs
        # while the query is embedded and the episodes are searched.
        pattern_task = None
        if want_facts and self._structural_head_enabled(use_adaptive_weights):
            pattern_task = asyncio.ensure_future(self._query_pattern(query))

        all_results: List[Dict[str, Any]] = []
        layer_errors: Dict[str, str] = {}
        facts_metadata: Dict[str, Any] = {}
        try:
            query_emb = await self.embeddings.embed_query(query)

            # Layers run one after the other: they share one database session,
            # and SQLAlchemy sessions do not allow concurrent operations.
            if want_episodes:
                try:
                    episodic = await self._retrieve_episodic_layer(
                        workspace_id=workspace_id, query=query, top_k=top_k, query_embedding=query_emb)
                    all_results.extend(episodic['results'])
                except Exception as e:
                    logger.exception("Episodic retrieval failed")
                    layer_errors['episodic'] = str(e)

            if want_facts:
                try:
                    facts = await self._retrieve_with_qkv_attention(
                        workspace_id=workspace_id,
                        query=query,
                        top_k=top_k,
                        focus_buffer=focus_buffer,
                        use_sparse_attention=use_sparse_attention,
                        use_adaptive_weights=use_adaptive_weights,
                        use_hierarchical_attention=use_hierarchical_attention,
                        query_embedding=query_emb,
                        query_pattern=pattern_task,
                        explain=debug_mode
                    )
                    all_results.extend(facts['results'])
                    facts_metadata = facts.get('metadata', {})
                except Exception as e:
                    logger.exception("Fact retrieval failed")
                    layer_errors['facts'] = str(e)
        finally:
            if pattern_task is not None and not pattern_task.done():
                pattern_task.cancel()

        if layer_errors and len(layer_errors) == int(want_facts) + int(want_episodes):
            raise RuntimeError(f"Retrieval failed in every requested layer: {layer_errors}")

        # Same scale everywhere (cosine, plus the GKU bonus for facts).
        ranked = sorted(all_results, key=lambda x: x.get('score', 0.0), reverse=True)

        metadata_extension: Dict[str, Any] = {"rl_reranking_applied": False}
        if use_rl_feedback and len(ranked) > 1:
            focus_buffer_state = None
            if focus_buffer:
                focus_buffer_state = {item_id: item.current_weight
                                      for item_id, item in focus_buffer.items.items()}
            try:
                rl_prediction = await self.rl_feedback_system.get_ranked_responses(
                    query=query,
                    candidate_responses=ranked,
                    focus_buffer_state=focus_buffer_state,
                    query_embedding=query_emb,
                    candidate_embeddings=facts_metadata.get('candidate_embeddings', {})
                )
                ranked = rl_prediction.ranked_responses
                metadata_extension = {
                    "rl_reranking_applied": bool(rl_prediction.exploration_factor or
                                                 getattr(self.rl_feedback_system, 'has_learned', False)),
                    "rl_confidence_scores": rl_prediction.confidence_scores,
                    "rl_exploration_factor": rl_prediction.exploration_factor,
                    "rl_model_performance": self.rl_feedback_system.get_model_performance_metrics()
                }
            except Exception as e:
                logger.warning(f"RL feedback system failed, falling back to original ranking: {e}")
                metadata_extension = {"rl_reranking_applied": False, "rl_error": str(e)}

        final_results = self._diversify(ranked, top_k)

        fact_results = [r for r in final_results if r.get('item_type') == 'aku']
        layer_distribution = {
            'episodic': len([r for r in final_results if r.get('layer') == 'episodic']),
            'aku': len(fact_results),
            # Facts are labelled 'attention_mesh' for backward compatibility.
            'attention_mesh': len(fact_results),
            # Distinct GKUs the returned facts came through (GKUs themselves
            # are never returned as items).
            'gku': len({r['metadata']['gku_id'] for r in fact_results if r['metadata'].get('gku_id')}),
        }

        result = {
            "results": final_results,
            "metadata": {
                "mode": mode,
                "layers_retrieved": layers,
                "total_results": len(final_results),
                "layer_distribution": layer_distribution,
                "active_nodes": len(facts_metadata.get('selected_gku_ids', [])),
                "selected_gku_ids": facts_metadata.get('selected_gku_ids', []),
                "head_weights_used": facts_metadata.get('head_weights_used'),
                "score_scale": "cosine similarity (+ up to %.2f GKU bonus for facts)" % GKU_CONTEXT_BOOST,
                "retrieval_method": "gku_heads_plus_direct_fact_search",
                **metadata_extension
            }
        }
        if layer_errors:
            result["metadata"]["layer_errors"] = layer_errors

        if debug_mode:
            debug_attention = facts_metadata.get('debug_attention')
            if 'explanation' in facts_metadata:
                result["metadata"]["explanation"] = facts_metadata['explanation']
            if self.debugger is None:
                logger.warning("Debug mode enabled but AttentionDebugger not available")
            elif debug_attention is not None:
                try:
                    result["debug_info"] = self.debugger.create_debug_info(
                        query=query,
                        gku_names=debug_attention['gku_names'],
                        attention_result=debug_attention
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate debug info: {e}")

        return result

    @staticmethod
    def _diversify(ranked: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Top-k with at most MAX_RESULTS_PER_SOURCE items from any one source
        episode, so a chunk and the facts extracted from it do not fill the
        list together. Deferred items fill remaining slots if there are not
        enough other results, so nothing relevant is dropped for diversity.
        """
        picked, deferred, per_source = [], [], {}
        for r in ranked:
            source = r.get('source_episode_id')
            if source is not None and per_source.get(source, 0) >= MAX_RESULTS_PER_SOURCE:
                deferred.append(r)
                continue
            if source is not None:
                per_source[source] = per_source.get(source, 0) + 1
            picked.append(r)
            if len(picked) == top_k:
                return picked
        return (picked + deferred)[:top_k]

    def _structural_head_enabled(self, use_adaptive_weights: bool) -> bool:
        weights = (self.adaptive_weighting.get_current_weights()
                   if use_adaptive_weights else DEFAULT_HEAD_WEIGHTS)
        return weights.get('structural', 0.0) > 0

    async def _query_pattern(self, query: str) -> Dict[str, Any]:
        """The query's pattern signature for the structural head; {} on failure."""
        try:
            pattern = await self.query_extractor.extract_query_graph(query)
            logger.debug(f"Extracted query pattern: {pattern.get('graphlets')}")
            return pattern or {}
        except Exception as e:
            logger.warning(f"Query pattern extraction failed, structural head gets no signal: {e}")
            return {}

    async def _retrieve_episodic_layer(
        self,
        workspace_id: str,
        query: str,
        top_k: int = 5,
        query_embedding: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Retrieve from the episodic memory layer. 'score' is cosine similarity."""
        query_emb = query_embedding if query_embedding is not None else await self.embeddings.embed_query(query)
        episodes = await self.storage.find_similar_episodes(
            workspace_id, query_emb, threshold=EPISODIC_MIN_SIMILARITY, limit=top_k)

        results = []
        for episode in episodes:
            results.append({
                "id": episode['id'],
                "content": episode['content'],
                "summary": episode.get('summary'),
                "similarity": episode['similarity'],
                "semantic_similarity": episode['similarity'],
                "score": episode['similarity'],
                "layer": "episodic",
                "item_type": "episode",
                "source_episode_id": episode['id'],
                "created_at": episode.get('created_at')
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return {
            "results": results[:top_k],
            "metadata": {"layer": "episodic", "result_count": len(results)}
        }

    async def _retrieve_with_qkv_attention(
        self,
        workspace_id: str,
        query: str,
        top_k: int = 5,
        focus_buffer: Optional[FocusBuffer] = None,
        use_sparse_attention: bool = True,
        use_adaptive_weights: bool = True,
        use_hierarchical_attention: bool = False,
        query_embedding: Optional[List[float]] = None,
        query_pattern: Any = None,
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Facts from the best GKUs (scored by the semantic, structural and
        temporal heads) plus a direct similarity search over all facts.

        Args:
            query_embedding: reuse an embedding computed by the caller
            query_pattern: a pattern dict, an awaitable resolving to one, or
                None to extract it here
            explain: also build a human-readable explanation and debug data
        """
        query_emb = query_embedding if query_embedding is not None else await self.embeddings.embed_query(query)
        if query_pattern is None:
            query_pattern = (await self._query_pattern(query)
                             if self._structural_head_enabled(use_adaptive_weights) else {})
        elif inspect.isawaitable(query_pattern):
            query_pattern = await query_pattern

        retriever = QKVAttentionRetriever(
            embedding_dim=len(query_emb),
            use_adaptive_weights=use_adaptive_weights,
            use_hierarchical_attention=use_hierarchical_attention,
            adaptive_weighting=self.adaptive_weighting
        )
        result = await retriever.retrieve(
            storage=self.storage,
            workspace_id=workspace_id,
            query_embedding=query_emb,
            query_pattern=query_pattern,
            query_time_context={'timestamp': datetime.now()},
            top_k=top_k,
            use_sparse_attention=use_sparse_attention,
            focus_buffer=focus_buffer
        )
        interp = result.interpretations
        gku_scores = interp.get('gku_scores', {})
        gku_info = interp.get('gku_info', {})

        # Facts of the selected GKUs, already fetched and scored by the retriever.
        candidates = list(interp.get('ranked_akus', []))
        seen_aku_ids = {aku['id'] for aku in candidates}

        # Direct fact search over the whole workspace. Without it, a fact is
        # only reachable if its GKU is among the top few, and before the first
        # consolidation (no GKUs yet) no fact is reachable at all.
        candidates.extend(await self._direct_fact_candidates(workspace_id, query_emb, top_k, seen_aku_ids))

        results = []
        embeddings_by_id: Dict[str, Any] = {}
        for aku in candidates:
            emb = aku.get('embedding')
            if isinstance(emb, str):
                emb = json.loads(emb)
            similarity = aku.get('aku_similarity')
            if similarity is None:
                similarity = self._cosine_similarity(query_emb, emb) if emb else 0.0

            gku_id = aku.get('parent_gku_id')
            gku_score = gku_scores.get(gku_id) if gku_id is not None else None
            context = gku_info.get(gku_id, {}) if gku_id is not None else {}
            bonus = GKU_CONTEXT_BOOST * min(max(gku_score, 0.0), 1.0) if gku_score is not None else 0.0
            if emb:
                embeddings_by_id[aku['id']] = emb

            results.append({
                "id": aku['id'],
                "content": aku['content'],
                "source_gku": context.get('name'),
                "layer": "attention_mesh",  # Keep layer name for compatibility
                "item_type": "aku",
                "found_by": "gku" if gku_id is not None else "direct_search",
                "score": similarity + bonus,
                "semantic_similarity": similarity,
                "gku_score": gku_score,
                "gku_bonus": bonus,
                "source_episode_id": aku.get('episode_id'),
                "metadata": {
                    "gku_id": gku_id,
                    "gku_name": context.get('name'),
                    "gku_description": context.get('description'),
                    "aku_original_metadata": aku.get('aku_metadata', {}),
                    "created_at": aku.get('created_at')
                },
                "relevance_breakdown": {
                    "semantic": similarity,
                    "gku_bonus": bonus
                }
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:top_k]

        metadata: Dict[str, Any] = {
            "selected_gku_ids": interp.get('selected_gku_ids', []),
            "head_weights_used": interp.get('head_weights_used'),
            "retrieval_method": "gku_heads_plus_direct_fact_search",
            # For the RL reranker, so it does not embed the facts' text again
            "candidate_embeddings": {r['id']: embeddings_by_id[r['id']] for r in results if r['id'] in embeddings_by_id},
        }
        if explain:
            interpretations = retriever.interpret_attention_weights(result)
            interpretations.pop('ranked_akus', None)
            from ..explainability import generate_retrieval_explanation
            try:
                metadata["explanation"] = generate_retrieval_explanation(
                    query=query, results=results,
                    attention_weights=result.weights, interpretations=interpretations)
            except Exception as e:
                logger.warning(f"Failed to generate retrieval explanation: {e}")
            metadata["debug_attention"] = {
                "gku_names": [gku_info.get(g, {}).get('name') or g for g in interp.get('selected_gku_ids', [])],
                "weights": result.weights,
                "interpretations": interpretations,
            }
        return {"results": results, "metadata": metadata}

    def record_head_feedback(
        self,
        performance: Dict[str, float],
        query_embedding: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Report how useful each head was (e.g. {'semantic': 0.9, 'structural': 0.2,
        'temporal': 0.5}, higher is better) and move the head weights toward it.
        Nothing calls this automatically: EKM has no per-head relevance signal
        yet, so until something reports one the configured weights are used.
        Returns the new weights.
        """
        embedding = np.asarray(query_embedding if query_embedding is not None else [0.0], dtype=float)
        return dict(self.adaptive_weighting.update_weights_based_on_query(
            embedding, None, None, performance
        ))

    async def _direct_fact_candidates(
        self,
        workspace_id: str,
        query_emb: List[float],
        top_k: int,
        exclude_ids: set
    ) -> List[Dict[str, Any]]:
        """
        The facts most similar to the query across the whole workspace, minus
        those already collected through GKUs. Each comes back without a
        parent GKU and with its embedding, so it is scored like the others.
        """
        try:
            matches = await self.storage.find_similar_akus(
                workspace_id, query_emb, threshold=0.0, limit=max(top_k * 3, 10)
            )
            missing = [m['id'] for m in matches if m.get('id') and m['id'] not in exclude_ids]
            if not missing:
                return []
            # find_similar_akus does not return embeddings, so load them.
            akus = await self.storage.get_akus_by_ids(missing)
        except Exception as e:
            logger.warning(f"Direct fact search failed, using GKU candidates only: {e}")
            return []

        candidates = []
        for aku in akus:
            if aku['id'] in exclude_ids:
                continue
            candidate = aku.copy()
            candidate['parent_gku_id'] = None
            candidates.append(candidate)
            exclude_ids.add(aku['id'])
        return candidates

    def _cosine_similarity(self, v1, v2):
        """Calculate cosine similarity between two vectors."""
        a = np.asarray(v1, dtype=float)
        b = np.asarray(v2, dtype=float)
        if a.shape != b.shape:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
