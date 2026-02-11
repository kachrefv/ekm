"""
QKV Attention Implementation for EKM - Implements multi-head attention mechanism
with semantic, structural, and temporal retrieval heads.
"""

import numpy as np
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

# Import torch conditionally
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    from ..state import FocusBuffer
    from ..scalability import ScalableEKMBackend
    from ..rl_feedback import RLFeedbackSystem
    from ...storage.base import BaseStorage
    from ...providers.base import BaseLLM, BaseEmbeddings
    from ..query_analysis import QueryGraphExtractor
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None
    # Fallback imports for non-torch environment if necessary, or handle gracefully
    # For now, assume these are only needed if torch is available or will be handled by other means
    # If these modules are truly independent of torch, they should be outside the try block.
    # Given the instruction, they are placed here.
    from ..state import FocusBuffer
    from ..scalability import ScalableEKMBackend
    from ..rl_feedback import RLFeedbackSystem
    from ...storage.base import BaseStorage
    from ...providers.base import BaseLLM, BaseEmbeddings
    from ..query_analysis import QueryGraphExtractor


from ..models import GKU, AKU
from ..graph import SparseGraphManager

logger = logging.getLogger(__name__)


@dataclass
class AttentionHeads:
    """Container for attention head results."""
    semantic: np.ndarray
    structural: np.ndarray
    temporal: np.ndarray


@dataclass
class AttentionResult:
    """Container for attention computation results."""
    scores: np.ndarray
    weights: Dict[str, np.ndarray]
    interpretations: Dict[str, Any]


class SparseAttention:
    """
    Sparse attention mechanism that uses specialized sparse matrix libraries (torch.sparse or scipy.sparse)
    to compute attention with O(N*k) complexity, avoiding the materialization of the full N^2 dense attention matrix.
    
    This implementation automatically selects the best available backend:
    1. PyTorch Sparse (if available and input is torch tensor)
    2. SciPy Sparse (if available and input is numpy array)
    3. Fallback to dense masked attention (if sparse libraries unavailable or graph structure is dense)
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, sparsity_factor: float = 0.1):
        """
        Initialize sparse attention.

        Args:
            d_model: Dimension of the model (embedding size)
            num_heads: Number of attention heads
            sparsity_factor: Fraction of connections to keep (0.0 to 1.0)
        """
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.sparsity_factor = sparsity_factor
        self.use_torch = TORCH_AVAILABLE

        if self.use_torch:
            # Initialize PyTorch parameters
            self.W_q = torch.nn.Linear(d_model, d_model)
            self.W_k = torch.nn.Linear(d_model, d_model)
            self.W_v = torch.nn.Linear(d_model, d_model)
            self.W_o = torch.nn.Linear(d_model, d_model)
        else:
            # Initialize NumPy parameters for fallback
            self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
            self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
            self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
            self.W_o = np.random.randn(d_model, d_model)
        
        # Graph manager to determine neighborhoods
        self.graph_manager = SparseGraphManager()

    def softmax(self, x: Any, axis: int = -1) -> Any:
        """Numerically stable softmax implementation."""
        if isinstance(x, np.ndarray):
            exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        else:
            return torch.softmax(x, dim=axis)

    def _torch_sparse_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        graph_structure: Optional[Any] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implement true sparse attention using PyTorch sparse operations.
        
        Args:
            Q, K, V: Tensors of shape (batch_size, seq_len, d_model)
            graph_structure: Sparse tensor or indices representing connectivity
        """
        batch_size, seq_len, _ = Q.shape
        
        # Split heads: (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # If no graph structure, we must compute full attention (or rely on local window)
        if graph_structure is None:
            # Fallback to standard scaled dot product attention (efficient implementation)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn, V)
            return output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model), attn
            
        # True sparse attention logic
        # Assuming graph_structure provides indices of allowed connections (adj matrix)
        # We only compute Q*K^T for these indices
        
        # This is a simplified implementation of BlockSparse or similar concepts
        # For full general sparsity, we calculate values only at specific indices
        
        # Check if graph_structure is a torch sparse tensor
        if not isinstance(graph_structure, torch.Tensor) or not graph_structure.is_sparse:
             # Try to convert if it's scipy sparse
            try:
                import scipy.sparse
                if scipy.sparse.issparse(graph_structure):
                    coo = graph_structure.tocoo()
                    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
                    values = torch.FloatTensor(coo.data)
                    shape = coo.shape
                    graph_structure = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
            except:
                pass

        if isinstance(graph_structure, torch.Tensor) and graph_structure.is_sparse:
            # Iterate over heads/batches to compute sparse attention
            # (Note: PyTorch sparse support for 4D batched mm is limited, may need loop)
            # This logic avoids creating N^2 dense matrix
            
            # Extract sparse indices
            indices = graph_structure._indices()
            rows, cols = indices[0], indices[1]
            
            # Compute Q[rows] dot K[cols]
            # Expanding to handle multi-head: (num_heads, num_edges)
            
            # Gather Q and K vectors for active edges
            # For simplicity, we process batch 0 (assuming batch=1 for knowledge graph retrieval)
            
            q_selected = Q[0, :, rows, :] # (num_heads, num_edges, d_k)
            k_selected = K[0, :, cols, :] # (num_heads, num_edges, d_k)
            
            # Dot product
            sparse_scores = (q_selected * k_selected).sum(dim=-1) / math.sqrt(self.d_k) # (num_heads, num_edges)
            
            # Sparse softmax is tricky. We need to normalize over 'rows' (source nodes)
            # Use torch_scatter or efficient loop if available, else naive loop over rows (slow in python)
            # For implementation plan, we use a numerically stable trick or fallback to masked if N is small enough
            # But here we committed to avoid N^2
            
            # "Poor man's sparse softmax":
            # 1. Exponentiate scores
            sparse_exp = torch.exp(sparse_scores - sparse_scores.max()) # Numerical stability
            
            # 2. Sum exp values per row
            # We use a dense tensor for row sums if N is manageable (N < 100k is fine for vector of size N)
            row_sums = torch.zeros((self.num_heads, seq_len), device=Q.device)
            row_sums.index_add_(1, rows, sparse_exp)
            
            # 3. Divide by sum
            rows_expanded = rows.unsqueeze(0).expand(self.num_heads, -1)
            sparse_attn_weights = sparse_exp / (row_sums.gather(1, rows_expanded) + 1e-6)
            
            # Compute V aggregation: Sparse * Dense -> Dense
            # Output[r] = sum(weight[r, c] * V[c])
            # Again, can use index_add_
            
            v_values = V[0, :, cols, :] # (num_heads, num_edges, d_k)
            weighted_v = v_values * sparse_attn_weights.unsqueeze(-1)
            
            output = torch.zeros((self.num_heads, seq_len, self.d_k), device=Q.device)
            output.index_add_(1, rows, weighted_v)
            
            # Reshape back to (batch, seq_len, d_model)
            output = output.transpose(0, 1).reshape(1, seq_len, self.d_model)
            
            # Return sparse weights approximation
            return output, sparse_attn_weights
            
        else:
             # If graph structure isn't sparse tensor, warn and fallback
             logger.warning("Graph structure provided to TorchSparseAttention is not a sparse tensor. Falling back to dense with mask.")
             
             if hasattr(graph_structure, 'shape'):
                 if isinstance(graph_structure, np.ndarray):
                     mask = torch.from_numpy(graph_structure).to(Q.device).float()
                 else:
                     mask = graph_structure.float()
             else:
                 mask = None
                 
             scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
             
             if mask is not None:
                 scores = scores.masked_fill(mask == 0, -1e9)
                 
             attn = torch.softmax(scores, dim=-1)
             output = torch.matmul(attn, V)
             return output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model), attn

    def _scipy_sparse_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        graph_structure: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implement sparse attention using SciPy sparse matrices.
        Avoids N^2 intermediate dense matrix.
        """
        import scipy.sparse
        
        batch_size, seq_len, _ = Q.shape
        
        # Handle heads: Manual loop over heads to avoid complex broadcasting with sparse
        outputs = []
        attentions = []
        
        # Split heads
        Q_heads = self.split_heads(Q) # (batch, num_heads, seq_len, d_k)
        K_heads = self.split_heads(K)
        V_heads = self.split_heads(V)
        
        # Use graph structure constraints
        # Assume graph_structure is scipy sparse matrix or numpy array
        if graph_structure is not None:
            if not scipy.sparse.issparse(graph_structure):
                graph_structure = scipy.sparse.csr_matrix(graph_structure)
            
            rows, cols = graph_structure.nonzero()
        else:
            # If no structure, fallback to dense (slow)
            return self.sparse_scaled_dot_product_attention(Q, K, V)

        # Process each head
        for h in range(self.num_heads):
             # For this head
            q_h = Q_heads[0, h] # (seq_len, d_k)
            k_h = K_heads[0, h] # (seq_len, d_k)
            v_h = V_heads[0, h] # (seq_len, d_k)
            
            # Compute dot product ONLY at non-zero indices
            # dot(Q[i], K[j]) for (i,j) in graph
            # We can use the property that sparse_matrix (N, N) with values can be built
            
            # Vectorized computation of sparse values
            # q_selected = q_h[rows]
            # k_selected = k_h[cols]
            # values = np.sum(q_selected * k_selected, axis=1) / sqrt(d_k)
            
            values = np.einsum('ij,ij->i', q_h[rows], k_h[cols]) / np.sqrt(self.d_k)
            
            # Create sparse matrix of logits
            sparse_logits = scipy.sparse.csr_matrix((values, (rows, cols)), shape=(seq_len, seq_len))
            
            # Sparse Softmax
            # exp(x - max) / sum(exp)
            # Row-wise max
            row_max = np.array(sparse_logits.max(axis=1).todense()).flatten()
            
            # We iterate to apply stable softmax efficiently on data array
            # indices of sparse_logits correspond to rows/cols
            # data = exp(data - row_max[row_indices])
            
            # Getting row indices for data matches
            # csr_matrix.indices gives cols, but we need rows corresponding to data
            # To do this efficiently, calculate exp directly on the data vector
            # We assume rows are aligned with standard COO/CSR traversal or re-extract
            coo = sparse_logits.tocoo() 
            
            # Numeric stability
            data_exp = np.exp(coo.data - row_max[coo.row])
            
            # Create new sparse matrix with exp values
            sparse_exp = scipy.sparse.csr_matrix((data_exp, (coo.row, coo.col)), shape=(seq_len, seq_len))
            
            # Row sums
            row_sums = np.array(sparse_exp.sum(axis=1)).flatten()
            
            # Normalize
            # Multiply sparse_exp by diagonal matrix of 1/sums
            inv_sums = np.reciprocal(row_sums, where=row_sums!=0)
            diagonal_inv = scipy.sparse.diags(inv_sums)
            
            attn_weights = diagonal_inv @ sparse_exp
            
            # Aggregate V
            # Output = Attn @ V (Sparse @ Dense -> Dense)
            # SciPy handles Sparse @ Dense efficiently
            head_output = attn_weights @ v_h
            
            outputs.append(head_output)
            attentions.append(attn_weights)
            
        # Stack heads
        # (num_heads, seq_len, d_k) -> (seq_len, num_heads, d_k) -> (seq_len, d_model)
        combined_output = np.stack(outputs, axis=1).reshape(batch_size, seq_len, self.d_model)
        
        return combined_output, attentions[0] # Return first head weights for interface compat

    def forward(
        self, 
        Q: Any, 
        K: Any, 
        V: Any, 
        graph_structure: Optional[Any] = None,
        mask: Optional[Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Forward pass dispatching to Torch or SciPy implementation.
        """
        if self.use_torch:
            # Ensure inputs are tensors
            if not isinstance(Q, torch.Tensor):
                Q = torch.tensor(Q, dtype=torch.float32)
            if not isinstance(K, torch.Tensor):
                K = torch.tensor(K, dtype=torch.float32)
            if not isinstance(V, torch.Tensor):
                V = torch.tensor(V, dtype=torch.float32)
                
            # Linear projections
            Q_proj = self.W_q(Q)
            K_proj = self.W_k(K)
            V_proj = self.W_v(V)
            
            output, weights = self._torch_sparse_attention(Q_proj, K_proj, V_proj, graph_structure)
            
            # Final projection
            output = self.W_o(output)
            
            # Convert back to numpy if input was numpy (simplifying assumption for interface)
            return output.detach().numpy(), weights
            
        else:
            # NumPy/SciPy path
            # Linear projections
            Q_proj = np.matmul(Q, self.W_q)
            K_proj = np.matmul(K, self.W_k)
            V_proj = np.matmul(V, self.W_v)
            
            try:
                # Try optimized sparse path first
                output, weights = self._scipy_sparse_attention(Q_proj, K_proj, V_proj, graph_structure)
            except Exception as e:
                logger.warning(f"SciPy sparse attention failed, falling back to dense: {e}")
                q_spl = self.split_heads(Q_proj)
                k_spl = self.split_heads(K_proj)
                v_spl = self.split_heads(V_proj)
                out_spl, weights = self.sparse_scaled_dot_product_attention(q_spl, k_spl, v_spl, graph_structure, mask)
                out_comb = np.transpose(out_spl, (0, 2, 1, 3)).reshape(Q.shape[0], Q.shape[1], self.d_model)
                output = np.matmul(out_comb, self.W_o)

            return output, weights

    # Keep original method for fallback/dense cases
    def sparse_scaled_dot_product_attention(
        self, 
        Q: np.ndarray, 
        K: np.ndarray, 
        V: np.ndarray, 
        graph_structure: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        # ... (original implementation kept for fallback correctness) ...
        # (Original body omitted here for brevity, conceptually it's the same as before)
        # For the tool call, I will include the existing method body to ensure it's not lost if I replace the class?
        # The prompt says replace the CLASS. So I must provide the full class content.
        # I'll paste the original method body below.

        # Handle the case where Q, K, V have been split into heads
        if len(Q.shape) == 4:  # (batch_size, num_heads, seq_len, depth)
            batch_size, num_heads, seq_len, d_model = Q.shape
        else:  # (batch_size, seq_len, d_model)
            batch_size, seq_len, d_model = Q.shape
            num_heads = 1
        
        # Compute attention scores: QK^T
        matmul_qk = np.matmul(Q, np.swapaxes(K, -1, -2))

        # Apply graph-based sparsity mask if provided
        if graph_structure is not None:
            if len(Q.shape) == 4:
                 # Check if graph_structure is compatible
                 if graph_structure.shape == (seq_len, seq_len):
                    expanded_graph = np.broadcast_to(graph_structure, (batch_size, num_heads, seq_len, seq_len))
                    matmul_qk = matmul_qk * expanded_graph
            else:
                if graph_structure.shape == (seq_len, seq_len):
                    expanded_graph = np.broadcast_to(graph_structure, (batch_size, seq_len, seq_len))
                    matmul_qk = matmul_qk * expanded_graph
            
            # Apply sparsity: keep only top-k connections per node
            k_sparse = max(1, int(seq_len * self.sparsity_factor))
            
            # Simple top-k masking (expensive dense op)
            for b in range(matmul_qk.shape[0]):
                for h in range(matmul_qk.shape[1] if len(matmul_qk.shape) == 4 else 1):
                    if len(matmul_qk.shape) == 4:
                        attn_matrix = matmul_qk[b, h]
                    else:
                        attn_matrix = matmul_qk[b]
                    
                    for i in range(attn_matrix.shape[0]):
                        attn_values = attn_matrix[i]
                        if len(attn_values) > k_sparse:
                            threshold = np.partition(attn_values, -k_sparse)[-k_sparse]
                            mask_val = (attn_values >= threshold)
                            if len(matmul_qk.shape) == 4:
                                matmul_qk[b, h, i] *= mask_val
                            else:
                                matmul_qk[b, i] *= mask_val

        dk = K.shape[-1]
        scaled_attention_logits = matmul_qk / np.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = self.softmax(scaled_attention_logits, axis=-1)

        output = np.matmul(attention_weights, V)
        return output, attention_weights



class MultiHeadAttention:
    """
    Multi-head attention mechanism for EKM retrieval.
    Implements the attention formula: Attention(Q, K, V) = softmax((QK^T)/√d)V
    """

    def __init__(self, d_model: int, num_heads: int = 8):
        """
        Initialize multi-head attention.

        Args:
            d_model: Dimension of the model (embedding size)
            num_heads: Number of attention heads
        """
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V for each head
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    
    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Queries (batch_size, seq_len, d_model)
            K: Keys (batch_size, seq_len, d_model)
            V: Values (batch_size, seq_len, d_model)
            mask: Optional mask to prevent attention to certain positions
        
        Returns:
            Attended values and attention weights
        """
        # Compute attention scores: QK^T
        matmul_qk = np.matmul(Q, np.swapaxes(K, -1, -2))
        
        # Scale by sqrt(d_k)
        dk = K.shape[-1]
        scaled_attention_logits = matmul_qk / np.sqrt(dk)
        
        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(scaled_attention_logits, axis=-1)
        
        # Multiply by values
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax implementation."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split the last dimension into (num_heads, depth).
        Returns shape: (batch_size, num_heads, seq_len, depth)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Reshape and transpose to separate heads
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return np.transpose(x, (0, 2, 1, 3))
    
    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Combine heads back to original shape.
        Input shape: (batch_size, num_heads, seq_len, depth)
        Output shape: (batch_size, seq_len, d_model)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        
        # Transpose and reshape
        x = np.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass of multi-head attention.
        
        Args:
            Q: Queries
            K: Keys  
            V: Values
            mask: Optional mask
        
        Returns:
            Attended output and attention weights for each head
        """
        batch_size = Q.shape[0]
        
        # Linear projections
        Q = np.matmul(Q, self.W_q)
        K = np.matmul(K, self.W_k)
        V = np.matmul(V, self.W_v)
        
        # Split into heads
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len_q, depth)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_len_k, depth)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_len_v, depth)
        
        # Scaled dot-product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape back to (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = np.transpose(scaled_attention, (0, 2, 1, 3))
        
        # Concatenate heads
        concat_attention = scaled_attention.reshape(batch_size, Q.shape[2], self.d_model)
        
        # Final linear projection
        output = np.matmul(concat_attention, self.W_o)
        
        return output, attention_weights


class SemanticRetrievalHead:
    """
    Semantic retrieval head that matches query embeddings with GKU content embeddings.
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
    
    def compute_attention(self, query_embedding: np.ndarray, gku_embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute semantic attention between query and GKU embeddings.
        
        Args:
            query_embedding: Query embedding vector
            gku_embeddings: List of GKU content embeddings
        
        Returns:
            Attention scores for each GKU
        """
        if len(gku_embeddings) == 0:
            return np.array([])
        
        # Stack GKU embeddings into a matrix
        gku_matrix = np.stack(gku_embeddings)
        
        # Compute cosine similarity between query and GKU embeddings
        query_norm = np.linalg.norm(query_embedding)
        gku_norms = np.linalg.norm(gku_matrix, axis=1)
        
        # Avoid division by zero
        query_norm = max(query_norm, 1e-8)
        gku_norms = np.maximum(gku_norms, 1e-8)
        
        # Compute dot products
        dot_products = np.dot(gku_matrix, query_embedding)
        
        # Compute cosine similarities
        similarities = dot_products / (query_norm * gku_norms)
        
        # Apply softmax to convert to attention weights
        exp_similarities = np.exp(similarities - np.max(similarities))  # Numerical stability
        attention_weights = exp_similarities / np.sum(exp_similarities)
        
        return attention_weights


class StructuralRetrievalHead:
    """
    Structural retrieval head that matches query pattern with GKU structural signatures.
    """
    
    def __init__(self):
        pass
    
    def compute_attention(self, query_pattern: Dict[str, Any], gku_patterns: List[Dict[str, Any]]) -> np.ndarray:
        """
        Compute structural attention between query pattern and GKU structural signatures.
        
        Args:
            query_pattern: Query pattern signature
            gku_patterns: List of GKU structural signatures
        
        Returns:
            Attention scores for each GKU
        """
        if len(gku_patterns) == 0:
            return np.array([])
        
        similarities = []
        
        for gku_pattern in gku_patterns:
            # Compute similarity between query pattern and GKU pattern
            similarity = self._compute_pattern_similarity(query_pattern, gku_pattern)
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        # Apply softmax to convert to attention weights
        exp_similarities = np.exp(similarities - np.max(similarities))  # Numerical stability
        attention_weights = exp_similarities / np.sum(exp_similarities)
        
        return attention_weights
    
    def _compute_pattern_similarity(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """
        Compute similarity between two pattern signatures using weighted combination.
        Aligned with ekm.core.patterns.calculate_pattern_similarity.
        """
        from ..core.patterns import calculate_pattern_similarity
        return calculate_pattern_similarity(pattern1, pattern2)


class TemporalRetrievalHead:
    """
    Temporal retrieval head that considers time-based relevance.
    """
    
    def __init__(self):
        pass
    
    def compute_attention(self, query_time_context: Optional[Dict[str, Any]], gku_times: List[Dict[str, Any]]) -> np.ndarray:
        """
        Compute temporal attention based on time relevance.
        
        Args:
            query_time_context: Time context of the query
            gku_times: List of GKU temporal metadata
        
        Returns:
            Attention scores for each GKU
        """
        if len(gku_times) == 0:
            return np.array([])
        
        # If no query time context, assign equal attention
        if query_time_context is None:
            return np.ones(len(gku_times)) / len(gku_times)
        
        # Extract query time info
        query_time = query_time_context.get('timestamp')
        if query_time is None:
            return np.ones(len(gku_times)) / len(gku_times)
        
        # Ensure query_time is a timestamp (float)
        if hasattr(query_time, 'timestamp'):
            query_time = query_time.timestamp()
        else:
            try:
                query_time = float(query_time)
            except (TypeError, ValueError):
                return np.ones(len(gku_times)) / len(gku_times)
        
        # Calculate temporal relevance for each GKU
        time_deltas = []
        for gku_time in gku_times:
            gku_created_at = gku_time.get('created_at')
            if gku_created_at is not None:
                # Calculate time difference
                gku_ts = gku_created_at.timestamp() if hasattr(gku_created_at, 'timestamp') else float(gku_created_at)
                delta = abs(query_time - gku_ts)
                time_deltas.append(delta)
            else:
                # If GKU has no timestamp, assign a large delta (less relevant)
                time_deltas.append(float('inf'))
        
        time_deltas = np.array(time_deltas)
        
        # Convert deltas to relevance scores (smaller delta = higher relevance)
        # Use exponential decay: relevance = exp(-delta / tau)
        # Where tau controls the decay rate
        tau = 86400 * 30  # Decay constant (30 days in seconds)
        
        # Handle infinite values
        finite_deltas = np.where(time_deltas == float('inf'), np.max(time_deltas[np.isfinite(time_deltas)]) * 2, time_deltas)
        
        # Calculate relevance scores
        relevance_scores = np.exp(-finite_deltas / tau)
        
        # Normalize to get attention weights
        exp_relevance = np.exp(relevance_scores - np.max(relevance_scores))  # Numerical stability
        attention_weights = exp_relevance / np.sum(exp_relevance)
        
        return attention_weights


class AdaptiveHeadWeighting:
    """
    Implements adaptive head weighting with meta-learning component that adjusts
    head weights based on query characteristics and historical performance.
    """
    
    def __init__(self, initial_weights: Dict[str, float] = None):
        """
        Initialize adaptive head weighting.

        Args:
            initial_weights: Initial head weights dictionary
        """
        if initial_weights is None:
            initial_weights = {
                'semantic': 0.4,
                'structural': 0.4,
                'temporal': 0.2
            }
        
        self.head_weights = initial_weights.copy()
        self.performance_history = {
            'semantic': [],
            'structural': [],
            'temporal': []
        }
        self.query_characteristics_history = []
        self.weight_adjustment_history = []
        
        # Meta-learning parameters
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.performance_decay = 0.95
        
    def update_weights_based_on_query(
        self, 
        query_embedding: np.ndarray, 
        query_pattern: Optional[Dict[str, Any]],
        query_time_context: Optional[Dict[str, Any]],
        performance_feedback: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Update head weights based on query characteristics and performance feedback.

        Args:
            query_embedding: Current query embedding
            query_pattern: Current query pattern
            query_time_context: Current query time context
            performance_feedback: Performance scores for each head

        Returns:
            Updated head weights
        """
        # Record query characteristics
        query_features = self._extract_query_features(query_embedding, query_pattern, query_time_context)
        self.query_characteristics_history.append({
            'features': query_features,
            'timestamp': time.time(),
            'performance': performance_feedback
        })
        
        # Update performance history with decay
        for head in self.performance_history:
            # Apply decay to previous performance scores
            decayed_scores = [score * self.performance_decay for score in self.performance_history[head]]
            # Add current performance score
            current_score = performance_feedback.get(head, 0.0)
            decayed_scores.append(current_score)
            self.performance_history[head] = decayed_scores[-100:]  # Keep last 100 scores
        
        # Adjust weights based on recent performance
        avg_performance = {
            head: np.mean(scores) if scores else 0.0 
            for head, scores in self.performance_history.items()
        }
        
        # Normalize performance scores to get weight adjustments
        total_performance = sum(avg_performance.values())
        if total_performance > 0:
            normalized_performance = {
                head: score / total_performance 
                for head, score in avg_performance.items()
            }
            
            # Apply momentum from previous adjustments
            if self.weight_adjustment_history:
                prev_weights = self.weight_adjustment_history[-1]
                for head in self.head_weights:
                    self.head_weights[head] = (
                        self.momentum * prev_weights[head] + 
                        (1 - self.momentum) * normalized_performance[head]
                    )
            else:
                self.head_weights = normalized_performance
        
        # Ensure weights sum to 1
        total_weight = sum(self.head_weights.values())
        if total_weight > 0:
            self.head_weights = {
                head: weight / total_weight 
                for head, weight in self.head_weights.items()
            }
        
        # Record current weight adjustment
        self.weight_adjustment_history.append(self.head_weights.copy())
        
        # Keep history sizes reasonable
        if len(self.query_characteristics_history) > 1000:
            self.query_characteristics_history = self.query_characteristics_history[-500:]
        
        return self.head_weights
    
    def _extract_query_features(
        self, 
        query_embedding: np.ndarray, 
        query_pattern: Optional[Dict[str, Any]], 
        query_time_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract features from query for meta-learning purposes.
        """
        features = {
            'embedding_magnitude': float(np.linalg.norm(query_embedding)),
            'embedding_variance': float(np.var(query_embedding)),
            'has_pattern': query_pattern is not None,
            'has_time_context': query_time_context is not None,
            'time_context_age': 0.0
        }
        
        if query_time_context and 'timestamp' in query_time_context:
            import datetime
            current_time = datetime.datetime.now().timestamp()
            query_time = query_time_context['timestamp'].timestamp() if hasattr(query_time_context['timestamp'], 'timestamp') else float(query_time_context['timestamp'])
            features['time_context_age'] = abs(current_time - query_time)
        
        return features
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current head weights."""
        return self.head_weights.copy()


class HierarchicalAttention:
    """
    Implements hierarchical attention mechanism with two stages:
    1. Coarse-grained attention to identify relevant GKU clusters
    2. Fine-grained attention within selected clusters
    """
    
    def __init__(self, d_model: int = 768, num_coarse_heads: int = 4, num_fine_heads: int = 8, 
                 coarse_top_k: int = 5, fine_top_k: int = 3):
        """
        Initialize hierarchical attention mechanism.

        Args:
            d_model: Model dimension
            num_coarse_heads: Number of attention heads for coarse stage
            num_fine_heads: Number of attention heads for fine stage
            coarse_top_k: Number of clusters to select in coarse stage
            fine_top_k: Number of items to select in fine stage
        """
        self.d_model = d_model
        self.num_coarse_heads = num_coarse_heads
        self.num_fine_heads = num_fine_heads
        self.coarse_top_k = coarse_top_k
        self.fine_top_k = fine_top_k
        
        # Coarse-grained attention for cluster selection
        self.coarse_attention = SparseAttention(
            d_model=d_model, 
            num_heads=num_coarse_heads, 
            sparsity_factor=0.3  # Higher sparsity for coarse attention
        )
        
        # Fine-grained attention for detailed selection within clusters
        self.fine_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_fine_heads
        )
        
        # Adaptive weighting for hierarchical attention
        self.adaptive_weighting = AdaptiveHeadWeighting({
            'coarse': 0.6,
            'fine': 0.4
        })

    def compute_coarse_attention(
        self,
        query_embedding: np.ndarray,
        cluster_representatives: List[np.ndarray],
        cluster_membership: List[List[int]],  # Maps cluster idx to member indices
        graph_structure: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Compute coarse-grained attention to identify relevant clusters.

        Args:
            query_embedding: Query embedding (d_model,)
            cluster_representatives: List of cluster representative embeddings
            cluster_membership: Mapping of cluster index to member indices
            graph_structure: Graph structure for sparse attention

        Returns:
            Tuple of (attention_scores, selected_cluster_indices)
        """
        if not cluster_representatives:
            return np.array([]), []
        
        # For coarse-grained attention, we'll use a simple similarity-based approach
        # since the attention mechanisms expect specific tensor shapes
        query_vec = query_embedding.reshape(1, -1)  # Shape: (1, d_model)
        cluster_matrix = np.array(cluster_representatives)  # Shape: (num_clusters, d_model)
        
        # Compute similarity scores between query and cluster representatives
        similarities = np.dot(cluster_matrix, query_vec.T).flatten()  # Shape: (num_clusters,)
        
        # Apply graph structure if provided (multiply by adjacency matrix weights)
        if graph_structure is not None and graph_structure.shape[0] == len(cluster_representatives):
            # Use graph structure to weight the similarities
            # This is a simplified approach - in practice, you might want to use more sophisticated graph attention
            weighted_similarities = similarities * np.mean(graph_structure, axis=1)  # Average connectivity
            cluster_scores = weighted_similarities
        else:
            cluster_scores = similarities
        
        # Select top-k clusters
        if len(cluster_scores) > 0:
            top_cluster_indices_full = np.argsort(cluster_scores)[-self.coarse_top_k:][::-1]
            # Ensure indices are within bounds and convert to list of ints
            top_cluster_indices = []
            for idx in top_cluster_indices_full:
                idx_val = int(idx) if np.isscalar(idx) or idx.ndim == 0 else int(idx.flatten()[0])
                if idx_val < len(cluster_representatives):
                    top_cluster_indices.append(idx_val)
        else:
            top_cluster_indices = []
        
        return cluster_scores, top_cluster_indices

    def compute_fine_grained_attention(
        self,
        query_embedding: np.ndarray,
        cluster_items: List[np.ndarray],  # Items within selected clusters
        cluster_item_indices: List[int]   # Original indices of items in global space
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Compute fine-grained attention within selected clusters.

        Args:
            query_embedding: Query embedding (d_model,)
            cluster_items: Items within selected clusters
            cluster_item_indices: Original indices of items in global space

        Returns:
            Tuple of (attention_scores, selected_item_indices)
        """
        if not cluster_items:
            return np.array([]), []
        
        # For fine-grained attention, we'll use a simple similarity-based approach
        query_vec = query_embedding.reshape(1, -1)  # Shape: (1, d_model)
        item_matrix = np.array(cluster_items)  # Shape: (num_items, d_model)
        
        # Compute similarity scores between query and cluster items
        similarities = np.dot(item_matrix, query_vec.T).flatten()  # Shape: (num_items,)
        item_scores = similarities
        
        # Select top-k items
        if len(item_scores) > 0 and len(cluster_item_indices) > 0:
            top_item_local_indices_full = np.argsort(item_scores)[-self.fine_top_k:][::-1]
            # Map local indices to global indices
            top_item_global_indices = []
            for i in top_item_local_indices_full:
                i_val = int(i) if np.isscalar(i) or i.ndim == 0 else int(i.flatten()[0])
                if i_val < len(cluster_item_indices):
                    idx = cluster_item_indices[i_val]
                    idx_val = int(idx) if np.isscalar(idx) or idx.ndim == 0 else int(idx.flatten()[0])
                    top_item_global_indices.append(idx_val)
        else:
            top_item_global_indices = []
        
        return item_scores, top_item_global_indices

    async def retrieve_hierarchical(
        self,
        storage: BaseStorage,
        workspace_id: str,
        query_embedding: List[float],
        query_pattern: Optional[Dict[str, Any]] = None,
        query_time_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        focus_buffer: Optional[Any] = None
    ) -> AttentionResult:
        """
        Perform hierarchical 2-stage attention retrieval:
        1. Attention-based GKU selection (influenced by focus buffer)
        2. Attention-based selection of AKUs belonging to those GKUs
        """
        # Step 1: Get GKUs and perform Stage 1 attention
        gkus = await storage.get_gkus_by_workspace(workspace_id)
        
        if not gkus:
            # Fallback: Mesh has not been consolidated yet. Retrieve directly from AKUs.
            akus = await storage.find_similar_akus(workspace_id, query_embedding, threshold=0.5, limit=top_k)
            if not akus:
                return AttentionResult(
                    scores=np.array([]),
                    weights={'hierarchical': np.array([])},
                    interpretations={'fallback_used': True, 'reason': 'no_gkus_or_akus'}
                )
            
            final_scores = [aku['similarity'] for aku in akus]
            interpretations = {
                'top_aku_ids': [aku['id'] for aku in akus],
                'hierarchical_2stage': False,
                'fallback_used': True,
                'reason': 'no_gkus_fresh_mesh'
            }
            
            return AttentionResult(
                scores=np.array(final_scores),
                weights={'aku_attention': np.array(final_scores)},
                interpretations=interpretations
            )
        
        # Prepare GKU data for Stage 1
        gku_embeddings = []
        gku_patterns = []
        gku_times = []
        gku_ids = []
        
        for gku in gkus:
            emb = gku.get('centroid_embedding')
            if emb is not None and len(emb) > 0:
                gku_embeddings.append(np.array(emb))
                gku_patterns.append(gku.get('pattern_signature', {}))
                gku_times.append({'created_at': gku.get('created_at')})
                gku_ids.append(gku['id'])
        
        if not gku_embeddings:
            return AttentionResult(scores=np.array([]), weights={}, interpretations={})

        query_emb_array = np.array(query_embedding)
        
        # Compute Stage 1 (GKU) attention
        # We'll use semantic, structural and temporal heads similar to the main retriever
        # but for simplicity in this method we'll use similarity + focus boost
        gku_matrix = np.stack(gku_embeddings)
        gku_sims = np.dot(gku_matrix, query_emb_array) / (np.linalg.norm(gku_matrix, axis=1) * np.linalg.norm(query_emb_array) + 1e-8)
        
        # Focus Buffer Boost for GKUs
        focus_boosts = np.zeros(len(gku_ids))
        if focus_buffer and hasattr(focus_buffer, 'items'):
            # In a real system, we'd have a mapping of AKU -> GKU. 
            # For now, we'll look for AKUs belonging to these GKUs that are in focus.
            for i, gku_id in enumerate(gku_ids):
                child_akus = await storage.get_akus_in_gku(gku_id)
                for child in child_akus:
                    if child['id'] in focus_buffer.items:
                        focus_boosts[i] += focus_buffer.items[child['id']].current_weight
        
        stage1_scores = gku_sims + 0.5 * focus_boosts
        top_gku_indices = np.argsort(stage1_scores)[-self.coarse_top_k:][::-1]
        selected_gku_ids = [gku_ids[i] for i in top_gku_indices]
        
        # Step 2: Get all AKUs from selected GKUs
        all_child_akus = []
        for gku_id in selected_gku_ids:
            akus_in_gku = await storage.get_akus_in_gku(gku_id)
            for aku in akus_in_gku:
                # Add parent GKU info for metadata
                aku['parent_gku_id'] = gku_id
                all_child_akus.append(aku)
        
        if not all_child_akus:
            return AttentionResult(scores=np.array([]), weights={}, interpretations={'selected_gku_ids': selected_gku_ids})
        
        # Step 3: Stage 2 Attention on Child AKUs
        aku_embeddings = [np.array(aku['embedding']) for aku in all_child_akus if aku.get('embedding')]
        if not aku_embeddings:
             return AttentionResult(scores=np.array([]), weights={}, interpretations={'selected_gku_ids': selected_gku_ids})
             
        aku_matrix = np.stack(aku_embeddings)
        aku_sims = np.dot(aku_matrix, query_emb_array) / (np.linalg.norm(aku_matrix, axis=1) * np.linalg.norm(query_emb_array) + 1e-8)
        
        # Final scores for AKUs
        top_aku_indices = np.argsort(aku_sims)[-top_k:][::-1]
        final_akus = [all_child_akus[i] for i in top_aku_indices]
        final_scores = [aku_sims[i] for i in top_aku_indices]
        
        # Prepare results
        interpretations = {
            'top_aku_ids': [aku['id'] for aku in final_akus],
            'selected_gku_ids': selected_gku_ids,
            'hierarchical_2stage': True,
            'gku_scores': {gku_ids[i]: float(stage1_scores[i]) for i in top_gku_indices}
        }
        
        return AttentionResult(
            scores=np.array(final_scores),
            weights={'aku_attention': aku_sims},
            interpretations=interpretations
        )


class QKVAttentionRetriever:
    """
    Main QKV Attention Retriever that combines all three heads.
    """

    def __init__(self, embedding_dim: int = 768, num_heads: int = 8, use_adaptive_weights: bool = False, use_hierarchical_attention: bool = False):
        self.semantic_head = SemanticRetrievalHead(embedding_dim)
        self.structural_head = StructuralRetrievalHead()
        self.temporal_head = TemporalRetrievalHead()
        self.multihead_attn = MultiHeadAttention(embedding_dim, num_heads)
        self.sparse_attn = SparseAttention(embedding_dim, num_heads)
        
        # Hierarchical attention
        self.use_hierarchical_attention = use_hierarchical_attention
        if use_hierarchical_attention:
            self.hierarchical_attn = HierarchicalAttention(
                d_model=embedding_dim,
                num_coarse_heads=max(1, num_heads//2),
                num_fine_heads=num_heads
            )
        
        # Option to use adaptive head weighting
        self.use_adaptive_weights = use_adaptive_weights
        if use_adaptive_weights:
            self.adaptive_weighting = AdaptiveHeadWeighting()
            # Start with initial weights
            self.head_weights = self.adaptive_weighting.get_current_weights()
        else:
            # Fixed head weights
            self.head_weights = {
                'semantic': 0.4,
                'structural': 0.4,
                'temporal': 0.2
            }
    
    async def retrieve(
        self,
        storage: BaseStorage,
        workspace_id: str,
        query_embedding: List[float],
        query_pattern: Optional[Dict[str, Any]] = None,
        query_time_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        use_sparse_attention: bool = False,
        focus_buffer: Optional[Any] = None
    ) -> AttentionResult:
        """
        Retrieve relevant AKUs using multi-head attention mechanism.

        Args:
            storage: Storage interface
            workspace_id: ID of the workspace to search in
            query_embedding: Embedding representation of the query
            query_pattern: Pattern signature of the query (optional)
            query_time_context: Time context of the query (optional)
            top_k: Number of top results to return
            use_sparse_attention: Whether to use sparse attention mechanism

        Returns:
            AttentionResult with scores, weights, and interpretations
        """
        try:
            # If using hierarchical attention, delegate to hierarchical attention mechanism
            if self.use_hierarchical_attention:
                return await self.hierarchical_attn.retrieve_hierarchical(
                    storage=storage,
                    workspace_id=workspace_id,
                    query_embedding=query_embedding,
                    query_pattern=query_pattern,
                    query_time_context=query_time_context,
                    top_k=top_k,
                    focus_buffer=focus_buffer
                )

            # Get all GKUs in the workspace
            gkus = await storage.get_gkus_by_workspace(workspace_id)

            if not gkus:
                return AttentionResult(
                    scores=np.array([]),
                    weights={'semantic': np.array([]), 'structural': np.array([]), 'temporal': np.array([])},
                    interpretations={}
                )

            # Prepare GKU data
            gku_embeddings = []
            gku_patterns = []
            gku_times = []
            gku_ids = []

            for gku in gkus:
                # Extract embeddings
                emb = gku.get('centroid_embedding')
                if emb is not None and len(emb) > 0:
                    gku_embeddings.append(np.array(emb))
                else:
                    # Fallback to zero embedding if missing, or skip
                    continue

                # Extract patterns
                gku_patterns.append(gku.get('pattern_signature', {}))

                # Extract temporal info
                gku_times.append({'created_at': gku.get('created_at')})
                gku_ids.append(gku['id'])

            # Convert query embedding to numpy array
            query_emb_array = np.array(query_embedding)

            # If using adaptive weights, update them based on query characteristics
            if self.use_adaptive_weights:
                # For now, we'll use a simple performance feedback mechanism
                performance_feedback = {
                    'semantic': 0.8,
                    'structural': 0.7,
                    'temporal': 0.6
                }
                
                self.head_weights = self.adaptive_weighting.update_weights_based_on_query(
                    query_emb_array,
                    query_pattern,
                    query_time_context,
                    performance_feedback
                )

            # Compute attention for each head
            semantic_scores = self.semantic_head.compute_attention(query_emb_array, gku_embeddings)
            structural_scores = self.structural_head.compute_attention(
                query_pattern or {},
                gku_patterns
            )
            temporal_scores = self.temporal_head.compute_attention(
                query_time_context,
                gku_times
            )

            # Normalize scores
            if len(semantic_scores) > 0:
                semantic_scores = semantic_scores / (np.sum(semantic_scores) + 1e-8)
            if len(structural_scores) > 0:
                structural_scores = structural_scores / (np.sum(structural_scores) + 1e-8)
            if len(temporal_scores) > 0:
                temporal_scores = temporal_scores / (np.sum(temporal_scores) + 1e-8)

            # Apply Focus Buffer boosts to GKUs
            focus_boosts = np.ones(len(gkus))
            if focus_buffer and hasattr(focus_buffer, 'items'):
                # For each GKU, check if its AKUs are in the focus buffer
                for i, gku in enumerate(gkus):
                    gku_id = gku['id']
                    # We might need to fetch AKUs for this GKU to check focus
                    # For performance, we'll assume the GKU model has a reference or we check recent activations
                    # Simplified: if any AKU in focus buffer belongs to this GKU
                    # In a real impl, we'd have a mapping. Let's assume a simplified boost for now.
                    for item_id, item in focus_buffer.items.items():
                        # If the focus item weight is high, boost related GKUs
                        # This logic would ideally be backed by a gku_id in the FocusItem or a lookup
                        pass

            # Combined scores (simplified - ignoring sparse attention for GKUs for now as they are fewer)
            combined_scores = (
                self.head_weights['semantic'] * semantic_scores +
                self.head_weights['structural'] * structural_scores +
                self.head_weights['temporal'] * temporal_scores
            )

            # Normalize combined scores
            combined_scores = combined_scores / (np.sum(combined_scores) + 1e-8)

            # Get top-k results
            if len(combined_scores) > 0:
                top_indices = np.argsort(combined_scores)[-top_k:][::-1]
                top_scores = combined_scores[top_indices]
            else:
                top_indices = np.array([])
                top_scores = np.array([])

            # Prepare interpretations
            if len(top_indices) > 0:
                top_indices_list = [int(idx) for idx in top_indices.flatten()] if isinstance(top_indices, np.ndarray) else [int(idx) for idx in top_indices]
                top_gku_ids = [gku_ids[i] for i in top_indices_list if i < len(gku_ids)]
                semantic_contributions = semantic_scores[top_indices] if len(semantic_scores) > 0 else []
                structural_contributions = structural_scores[top_indices] if len(structural_scores) > 0 else []
                temporal_contributions = temporal_scores[top_indices] if len(temporal_scores) > 0 else []
            else:
                top_gku_ids = []
                semantic_contributions = []
                structural_contributions = []
                temporal_contributions = []

            interpretations = {
                'top_gku_ids': top_gku_ids,
                'semantic_contributions': semantic_contributions,
                'structural_contributions': structural_contributions,
                'temporal_contributions': temporal_contributions,
                'head_weights_used': self.head_weights,
                'use_sparse_attention': use_sparse_attention,
                'adaptive_weights_used': self.use_adaptive_weights,
                'hierarchical_attention_used': self.use_hierarchical_attention
            }

            # Return attention weights for each head
            weights = {
                'semantic': semantic_scores,
                'structural': structural_scores,
                'temporal': temporal_scores
            }

            return AttentionResult(
                scores=top_scores,
                weights=weights,
                interpretations=interpretations
            )

        except Exception as e:
            logger.error(f"Error in QKV attention retrieval: {e}")
            raise
    
    def interpret_attention_weights(self, result: AttentionResult) -> Dict[str, Any]:
        """
        Interpret the attention weights to provide explanations for retrieval decisions.
        
        Args:
            result: AttentionResult from retrieval
        
        Returns:
            Dictionary with interpretations
        """
        interpretations = result.interpretations.copy()
        
        # Add more detailed interpretations
        if 'semantic_contributions' in interpretations and 'structural_contributions' in interpretations:
            sem_contr = interpretations['semantic_contributions']
            struct_contr = interpretations['structural_contributions']
            temp_contr = interpretations.get('temporal_contributions', [])
            
            if len(sem_contr) > 0:
                interpretations['primary_relevance_factor'] = 'semantic' if np.mean(sem_contr) > np.mean(struct_contr) else 'structural'
            
            # Calculate contribution percentages
            if len(sem_contr) > 0:
                interpretations['average_semantic_contribution'] = float(np.mean(sem_contr))
                interpretations['average_structural_contribution'] = float(np.mean(struct_contr))
                if len(temp_contr) > 0:
                    interpretations['average_temporal_contribution'] = float(np.mean(temp_contr))
        
        return interpretations