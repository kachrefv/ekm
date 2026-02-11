import asyncio
from typing import List, Dict, Any, Optional
from .state import FocusBuffer
from .scalability import ScalableEKMBackend
from .advanced_embeddings import MultiModalEmbedder, EnhancedEmbeddingProvider
from .rl_feedback import RLFeedbackSystem
from .training import TrainingService
from .attention.mechanisms import MultiHeadAttention, SparseAttention
from .graph.operations import SparseGraphManager
from .knowledge.extraction import KnowledgeExtractor
from .knowledge.organization import apply_spectral_clustering
from .retrieval.service import RetrievalService
from ..storage.base import BaseStorage
from ..providers.base import BaseLLM, BaseEmbeddings
import logging

logger = logging.getLogger(__name__)

class EKM:
    """Main Episodic Knowledge Mesh implementation - Orchestrator class."""

    def __init__(
        self,
        storage: BaseStorage,
        llm: BaseLLM,
        embeddings: BaseEmbeddings,
        config: Optional[Dict[str, Any]] = None
    ):
        self.storage = storage
        self.llm = llm
        self.embeddings = embeddings
        self.config = config or {}

        # Initialize scalable backend by default
        vector_dim = self.config.get("VECTOR_DIMENSION", None)  # None = auto-detect from embeddings
        cache_size = self.config.get("CACHE_SIZE", 10000)
        max_graph_nodes = self.config.get("MAX_GRAPH_NODES", 100000)

        self.scalable_backend = ScalableEKMBackend(
            storage=storage,
            embeddings=embeddings,
            vector_dim=vector_dim,
            cache_size=cache_size,
            max_graph_nodes=max_graph_nodes
        )

        # Default settings as per paper specifications
        self.min_chunk_size = self.config.get("EKM_MIN_CHUNK_SIZE", 512)  # As specified in paper
        self.max_chunk_size = self.config.get("EKM_MAX_CHUNK_SIZE", 2048)  # As specified in paper
        self.semantic_threshold = self.config.get("EKM_SEMANTIC_THRESHOLD", 0.82)

        # Initialize enhanced embedding provider if available
        try:
            # Check if we should use enhanced embeddings based on config
            use_enhanced_embeddings = self.config.get("USE_ENHANCED_EMBEDDINGS", False)
            if use_enhanced_embeddings:
                domain_model = self.config.get("DOMAIN_MODEL_NAME", None)
                self.enhanced_embedder = MultiModalEmbedder(
                    model_name=self.config.get("BASE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                    domain_model_name=domain_model
                )
                self.embeddings = EnhancedEmbeddingProvider(self.enhanced_embedder)
                # Update vector dimension based on enhanced embedder
                self.config["VECTOR_DIMENSION"] = self.enhanced_embedder.embedding_dim
                logger.info(f"Using enhanced multi-modal embeddings with dimension: {self.enhanced_embedder.embedding_dim}")
            else:
                self.enhanced_embedder = None
        except ImportError:
            logger.warning("Sentence Transformers not available, using original embeddings")
            self.enhanced_embedder = None

        # Initialize services
        self.training_service = TrainingService(
            storage=storage,
            llm=llm,
            embeddings=embeddings,
            scalable_backend=self.scalable_backend,
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size,
            semantic_threshold=self.semantic_threshold
        )
        
        self.rl_feedback_system = RLFeedbackSystem(
            storage=storage,
            llm=llm,
            embeddings=embeddings,
            learning_rate=self.config.get("RL_LEARNING_RATE", 0.1),
            exploration_rate=self.config.get("RL_EXPLORATION_RATE", 0.1),
            memory_size=self.config.get("RL_MEMORY_SIZE", 1000)
        )
        
        self.retrieval_service = RetrievalService(
            storage=storage,
            llm=llm,
            embeddings=embeddings,
            scalable_backend=self.scalable_backend,
            rl_feedback_system=self.rl_feedback_system,
            semantic_threshold=self.semantic_threshold
        )
        
        logger.info("Initialized EKM with modular services for training, retrieval, and feedback")

    async def train(self, workspace_id: str, text: str, title: Optional[str] = None):
        """Train the EKM with new text data using single-step extraction as described in the paper."""
        return await self.training_service.train(workspace_id, text, title)

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
        mode: str = 'hybrid'
    ) -> Dict[str, Any]:
        """Retrieve relevant knowledge from the mesh using unified QKV attention-based mechanism as described in the paper.
        
        Args:
            workspace_id: Workspace to retrieve from
            query: Query string
            top_k: Number of results to return
            focus_buffer: Optional focus buffer for context
            use_rl_feedback: Enable RL-based reranking
            use_sparse_attention: Use sparse attention for efficiency
            use_adaptive_weights: Use adaptive head weighting
            use_hierarchical_attention: Use hierarchical attention
            layers: Which layers to retrieve from ('episodic', 'aku', 'gku')
            debug_mode: If True, include detailed attention debug info in response
            mode: Retrieval mode ('hybrid', 'episodic', or 'causal')
        
        Returns:
            Dict with 'results' and 'metadata'. If debug_mode=True, also includes 'debug_info'.
        """
        return await self.retrieval_service.retrieve(
            workspace_id=workspace_id,
            query=query,
            top_k=top_k,
            focus_buffer=focus_buffer,
            use_rl_feedback=use_rl_feedback,
            use_sparse_attention=use_sparse_attention,
            use_adaptive_weights=use_adaptive_weights,
            use_hierarchical_attention=use_hierarchical_attention,
            layers=layers,
            debug_mode=debug_mode,
            mode=mode
        )

    async def record_response_feedback(
        self,
        query: str,
        response_id: str,
        response_content: str,
        user_rating: float,  # -1 (bad) to 1 (good)
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record user feedback for a specific query-response pair to improve
        the RL model over time.
        """
        return await self.rl_feedback_system.record_feedback(
            query=query,
            response_id=response_id,
            response_content=response_content,
            user_rating=user_rating,
            context=context
        )

    async def get_rl_model_performance(self) -> Dict[str, float]:
        """
        Get performance metrics for the RL feedback system.
        """
        return self.rl_feedback_system.get_model_performance_metrics()

    async def batch_record_feedback(
        self,
        query: str,
        responses_with_ratings: List[tuple]  # (response_id, content, rating)
    ) -> bool:
        """
        Record feedback for multiple responses to the same query.
        """
        return await self.rl_feedback_system.batch_record_feedback(
            query=query,
            responses_with_ratings=responses_with_ratings
        )