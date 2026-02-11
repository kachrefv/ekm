"""
Reinforcement Learning Feedback System for EKM
Implements a reward-based learning system to improve response ranking and relevance
"""
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid
import logging
from ..storage.base import BaseStorage
from ..providers.base import BaseLLM, BaseEmbeddings

logger = logging.getLogger(__name__)

@dataclass
class FeedbackRecord:
    """Record of user feedback for a specific query-response pair"""
    query: str
    response_id: str
    response_content: str
    user_rating: float  # Rating from -1 (bad) to 1 (good)
    timestamp: datetime
    query_embedding: List[float]
    response_embedding: List[float]
    context: Dict[str, Any]  # Additional context like focus buffer state, etc.

@dataclass
class RLState:
    """Represents the current state for RL decision making"""
    query_embedding: List[float]
    available_responses: List[Dict[str, Any]]
    focus_buffer_state: Optional[Dict[str, float]] = None
    historical_preferences: Optional[Dict[str, float]] = None

@dataclass
class RLPrediction:
    """Prediction from the RL model"""
    ranked_responses: List[Dict[str, Any]]
    confidence_scores: List[float]
    exploration_factor: float

class RLFeedbackSystem:
    """
    Reinforcement Learning system that learns from user feedback to improve
    response ranking and relevance in EKM queries.
    """
    
    MODEL_ID = "global"  # Default model identifier for persistence
    
    def __init__(
        self,
        storage: BaseStorage,
        llm: BaseLLM,
        embeddings: BaseEmbeddings,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 0.1,
        memory_size: int = 1000,
        model_id: str = None
    ):
        self.storage = storage
        self.llm = llm
        self.embeddings = embeddings
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.memory_size = memory_size
        self.model_id = model_id or self.MODEL_ID
        
        # Maintain a memory of feedback records
        self.feedback_memory: List[FeedbackRecord] = []
        
        # Feature names for interpretability
        self.feature_names = [
            'semantic_similarity',
            'focus_buffer_boost',
            'recency_score',
            'frequency_score',
            'confidence_score',
            'diversity_score',
            'length_score',
            'complexity_score',
            'relevance_score',
            'contextual_score'
        ]
        
        # Model parameters - will be loaded from storage or initialized
        # Initialize with deterministic weights favoring semantic similarity
        # self.weights = np.random.normal(0, 0.1, size=(len(self.feature_names),))
        self.weights = np.zeros(len(self.feature_names))
        self.weights[0] = 1.0  # Semantic similarity is the primary signal
        # Other weights initialized to 0.0 or small values for cold start
        
        self._weights_loaded = False

    async def load_weights(self):
        """Load model weights from persistent storage if available."""
        if self._weights_loaded:
            return
        
        try:
            rl_state = await self.storage.load_rl_state(self.model_id)
            if rl_state and rl_state.get('weights'):
                loaded_weights = np.array(rl_state['weights'])
                # Ensure weights match current feature count
                if len(loaded_weights) == len(self.feature_names):
                    self.weights = loaded_weights
                    logger.info(f"Loaded RL weights for model '{self.model_id}' (last updated: {rl_state.get('updated_at')})")
                else:
                    logger.warning(f"Loaded weights size mismatch ({len(loaded_weights)} vs {len(self.feature_names)}), using random init")
            self._weights_loaded = True
        except Exception as e:
            logger.warning(f"Could not load RL weights: {e}")
            self._weights_loaded = True

    async def save_weights(self):
        """Persist model weights to storage."""
        try:
            metadata = {
                'feature_names': self.feature_names,
                'learning_rate': self.learning_rate,
                'feedback_count': len(self.feedback_memory)
            }
            await self.storage.save_rl_state(
                model_id=self.model_id,
                weights=self.weights.tolist(),
                metadata=metadata
            )
            logger.debug(f"Saved RL weights for model '{self.model_id}'")
        except Exception as e:
            logger.warning(f"Could not save RL weights: {e}")
    
    async def record_feedback(
        self,
        query: str,
        response_id: str,
        response_content: str,
        user_rating: float,  # -1 to 1
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record user feedback for a query-response pair.
        
        Args:
            query: The original query
            response_id: ID of the response
            response_content: Content of the response
            user_rating: Rating from -1 (very bad) to 1 (very good)
            context: Additional context information
        
        Returns:
            True if feedback was recorded successfully
        """
        try:
            # Get embeddings for the query and response
            query_embedding = await self.embeddings.embed_query(query)
            response_embedding = await self.embeddings.embed_query(response_content)
            
            # Create feedback record
            feedback_record = FeedbackRecord(
                query=query,
                response_id=response_id,
                response_content=response_content,
                user_rating=user_rating,
                timestamp=datetime.now(),
                query_embedding=query_embedding,
                response_embedding=response_embedding,
                context=context or {}
            )
            
            # Add to memory
            self.feedback_memory.append(feedback_record)
            
            # Maintain memory size
            if len(self.feedback_memory) > self.memory_size:
                self.feedback_memory.pop(0)  # Remove oldest
            
            # Update model based on feedback
            await self._update_model(feedback_record)
            
            logger.info(f"Recorded feedback for response {response_id}, rating: {user_rating}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return False
    
    async def get_ranked_responses(
        self,
        query: str,
        candidate_responses: List[Dict[str, Any]],
        focus_buffer_state: Optional[Dict[str, float]] = None
    ) -> RLPrediction:
        """
        Get ranked responses using the RL model with potential exploration.
        
        Args:
            query: The query to rank responses for
            candidate_responses: List of candidate responses with metadata
            focus_buffer_state: Current state of focus buffer
        
        Returns:
            RLPrediction with ranked responses and confidence scores
        """
        if not self._weights_loaded:
             await self.load_weights()

        query_embedding = await self.embeddings.embed_query(query)
        
        # Prepare RL state
        rl_state = RLState(
            query_embedding=query_embedding,
            available_responses=candidate_responses,
            focus_buffer_state=focus_buffer_state
        )
        
        # Extract features for each response
        features_matrix = []
        for response in candidate_responses:
            features = await self._extract_features(rl_state, response)
            features_matrix.append(features)
        
        features_matrix = np.array(features_matrix)
        
        # Calculate scores using current model weights
        scores = np.dot(features_matrix, self.weights)
        
        # Apply exploration if needed (epsilon-greedy)
        exploration_factor = self.exploration_rate
        if np.random.random() < self.exploration_rate:
            # Add some random noise for exploration
            exploration_noise = np.random.normal(0, 0.1, size=scores.shape)
            scores += exploration_noise
            exploration_factor = 1.0  # Indicate exploration happened
        else:
            exploration_factor = 0.0
        
        # Rank responses by scores (highest first)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_responses = [candidate_responses[i] for i in ranked_indices]
        confidence_scores = scores[ranked_indices].tolist()
        
        return RLPrediction(
            ranked_responses=ranked_responses,
            confidence_scores=confidence_scores,
            exploration_factor=exploration_factor
        )
    
    async def _extract_features(
        self,
        rl_state: RLState,
        response: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract features for a query-response pair.
        
        Args:
            rl_state: Current RL state
            response: Response to extract features for
        
        Returns:
            Feature vector as numpy array
        """
        features = np.zeros(len(self.feature_names))
        
        # 1. Semantic similarity (cosine similarity between query and response)
        query_emb = np.array(rl_state.query_embedding)
        resp_emb = np.array(response.get('embedding', await self.embeddings.embed_query(response.get('content', ''))))
        
        cos_sim = self._cosine_similarity(query_emb, resp_emb)
        features[0] = cos_sim
        
        # 2. Focus buffer boost
        focus_boost = 0.0
        if rl_state.focus_buffer_state and response.get('id') in rl_state.focus_buffer_state:
            focus_boost = rl_state.focus_buffer_state[response['id']]
        features[1] = focus_boost
        
        # 3. Recency score (if available)
        recency = response.get('recency_score', 0.5)  # Default to medium recency
        features[2] = recency
        
        # 4. Frequency score (if available)
        frequency = response.get('frequency_score', 0.5)
        features[3] = frequency
        
        # 5. Confidence score from original retrieval
        confidence = response.get('score', response.get('semantic_similarity', 0.5))
        features[4] = confidence
        
        # 6. Diversity score (compared to other responses)
        diversity = self._calculate_diversity_score(response, rl_state.available_responses)
        features[5] = diversity
        
        # 7. Length score (normalized)
        content_length = len(response.get('content', ''))
        length_score = min(content_length / 1000, 1.0)  # Normalize to 0-1
        features[6] = length_score
        
        # 8. Complexity score (based on content structure)
        complexity = self._estimate_complexity(response.get('content', ''))
        features[7] = complexity
        
        # 9. Relevance score (from original retrieval)
        relevance = response.get('relevance_score', 0.5)
        features[8] = relevance
        
        # 10. Contextual score (based on query-response fit)
        contextual = self._calculate_contextual_score(rl_state.query_embedding, resp_emb)
        features[9] = contextual
        
        return features
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm_v1 * norm_v2))
    
    def _calculate_diversity_score(self, response: Dict[str, Any], all_responses: List[Dict[str, Any]]) -> float:
        """Calculate how diverse this response is compared to others."""
        if len(all_responses) <= 1:
            return 1.0  # Fully diverse if it's the only response
        
        resp_emb = np.array(response.get('embedding', []))
        if len(resp_emb) == 0:
            return 0.5  # Neutral if no embedding
        
        # Calculate similarity to other responses
        similarities = []
        for other_resp in all_responses:
            if other_resp.get('id') != response.get('id'):
                other_emb = np.array(other_resp.get('embedding', []))
                if len(other_emb) > 0:
                    sim = self._cosine_similarity(resp_emb, other_emb)
                    similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity  # More diverse = lower similarity
    
    def _estimate_complexity(self, content: str) -> float:
        """Estimate complexity of content based on various factors."""
        if not content:
            return 0.0
        
        # Simple complexity measure based on sentence structure
        sentences = content.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Vocabulary complexity (simplified)
        words = content.lower().split()
        unique_words = len(set(words))
        vocab_richness = unique_words / len(words) if words else 0
        
        # Combine measures (normalized)
        complexity = (min(avg_sentence_length / 20, 1.0) + vocab_richness) / 2
        return min(complexity, 1.0)
    
    def _calculate_contextual_score(self, query_emb: np.ndarray, resp_emb: np.ndarray) -> float:
        """Calculate contextual fit between query and response."""
        # This could be enhanced with more sophisticated contextual analysis
        return self._cosine_similarity(query_emb, resp_emb)
    
    async def _update_model(self, feedback_record: FeedbackRecord):
        """
        Update the model weights based on feedback using gradient descent.
        """
        try:
            # Find responses that were presented for this query in history
            # For simplicity, we'll update based on the single feedback received
            # In a more advanced system, we'd use counterfactual reasoning
            
            # For now, we'll use a simple approach: update weights based on
            # the difference between predicted and actual rating
            response = {
                'id': feedback_record.response_id,
                'content': feedback_record.response_content,
                'embedding': feedback_record.response_embedding
            }
            
            # Create a dummy state to extract features
            rl_state = RLState(
                query_embedding=feedback_record.query_embedding,
                available_responses=[response]
            )
            
            features = await self._extract_features(rl_state, response)
            
            # Predicted score using current weights
            predicted_score = np.dot(features, self.weights)
            
            # Actual score is the user rating (-1 to 1)
            actual_score = feedback_record.user_rating
            
            # Calculate error
            error = actual_score - predicted_score
            
            # Update weights using gradient descent
            # Gradient is just the features vector for linear model
            self.weights += self.learning_rate * error * features
            
            # Clip weights to prevent explosion
            self.weights = np.clip(self.weights, -10, 10)
            
            # Persist updated weights to storage
            await self.save_weights()
            
            logger.debug(f"Updated model weights. Error: {error:.3f}, Rating: {actual_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
    
    async def get_personalized_weights(self, user_id: str) -> Optional[np.ndarray]:
        """
        Get personalized model weights for a specific user if available.
        """
        # In a full implementation, this would load user-specific weights
        # For now, return the global weights
        return self.weights
    
    async def batch_record_feedback(
        self,
        query: str,
        responses_with_ratings: List[Tuple[str, str, float]]  # (response_id, content, rating)
    ) -> bool:
        """
        Record feedback for multiple responses to the same query.
        
        Args:
            query: The original query
            responses_with_ratings: List of (response_id, content, rating) tuples
        
        Returns:
            True if all feedback was recorded successfully
        """
        success_count = 0
        for response_id, content, rating in responses_with_ratings:
            success = await self.record_feedback(query, response_id, content, rating)
            if success:
                success_count += 1
        
        logger.info(f"Batch recorded {success_count}/{len(responses_with_ratings)} feedback records")
        return success_count == len(responses_with_ratings)
    
    def get_model_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the RL model.
        """
        if not self.feedback_memory:
            return {"feedback_records": 0}
        
        # Calculate average rating
        ratings = [record.user_rating for record in self.feedback_memory]
        avg_rating = np.mean(ratings)
        
        # Calculate rating variance
        rating_variance = np.var(ratings)
        
        # Calculate most recent feedback trend
        recent_feedback = self.feedback_memory[-10:] if len(self.feedback_memory) >= 10 else self.feedback_memory
        recent_ratings = [record.user_rating for record in recent_feedback]
        recent_avg = np.mean(recent_ratings) if recent_ratings else 0.0
        
        return {
            "feedback_records": len(self.feedback_memory),
            "average_rating": float(avg_rating),
            "rating_variance": float(rating_variance),
            "recent_average_rating": float(recent_avg),
            "exploration_rate": self.exploration_rate
        }