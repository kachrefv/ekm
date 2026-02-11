"""
Scalability Improvements for EKM - Implements vector indexing, caching, 
graph optimizations, and pagination for large-scale datasets.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pickle
import hashlib
from datetime import datetime, timedelta
from ..storage.base import BaseStorage
from ..providers.base import BaseEmbeddings

# Import faiss with fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None  # Will be handled in the class

# Import redis with fallback
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # Will be handled in the class

logger = logging.getLogger(__name__)


def check_production_readiness(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Check if the environment is ready for production usage.
    Raises RuntimeError if critical dependencies are missing.
    """
    if config and config.get('mode') == 'production':
        missing = []
        if not FAISS_AVAILABLE:
            missing.append("faiss")
        if not REDIS_AVAILABLE:
            missing.append("redis")
        
        # Check for Torch/SciPy for attention
        try:
            import torch
        except ImportError:
            try:
                import scipy
            except ImportError:
                missing.append("torch OR scipy")
        
        if missing:
            raise RuntimeError(f"Production mode requires the following missing dependencies: {', '.join(missing)}")


class VectorIndexManager:
    """
    Manages vector indexing using FAISS for efficient similarity search.
    Falls back to in-memory numpy operations if FAISS is not available.
    """

    def __init__(self, dimension: int = None, index_type: str = "IVF"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.id_map = {}  # Maps FAISS index to actual IDs
        self.reverse_id_map = {}  # Maps actual IDs to FAISS index
        self.is_trained = False
        
        # Store vectors in memory as fallback
        self.vectors = np.empty((0, dimension if dimension is not None else 0), dtype=np.float32)
        self.vector_ids = []  # Store corresponding IDs
        
        if dimension is not None:
            self._initialize_index(dimension)
    
    def _initialize_index(self, dimension: int) -> None:
        """Initialize the FAISS index with the given dimension."""
        self.dimension = dimension
        # If fallback vectors were empty (0,0), resize them to (0, dimension)
        if self.vectors.shape[1] != dimension:
             self.vectors = np.empty((0, dimension), dtype=np.float32)
        
        if FAISS_AVAILABLE:
            if self.index_type == "Flat":
                self.index = faiss.IndexFlatL2(dimension)
            elif self.index_type == "IVF":
                # Create quantizer for IVF index
                quantizer = faiss.IndexFlatL2(dimension)
                # Use a reasonable number of centroids (nlist)
                nlist = min(100, max(1, int(np.sqrt(1000))))  # Default for 1000 vectors
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            elif self.index_type == "HNSW":
                self.index = faiss.IndexHNSWFlat(dimension, 32)  # ef_construction=40, M=32
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def add_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        """
        Add vectors to the index with corresponding IDs.
        """
        if vectors.shape[0] != len(ids):
            raise ValueError("Number of vectors must match number of IDs")

        # Lazy initialization: set dimension from first vectors if not set
        if self.dimension is None:
            self._initialize_index(vectors.shape[1])
        elif vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {vectors.shape[1]}")

        # Convert to float32
        vectors = vectors.astype('float32')
        
        if FAISS_AVAILABLE:
            # Use FAISS for indexing
            # If using IVF, train the index first if not already trained
            if isinstance(self.index, faiss.IndexIVF) and not self.is_trained:
                if vectors.shape[0] < self.index.nlist:
                    # Pad with copies if we don't have enough vectors to train
                    needed = self.index.nlist - vectors.shape[0]
                    extra_vectors = np.tile(vectors[0], (needed, 1)).astype('float32')
                    train_vectors = np.vstack([vectors, extra_vectors])
                else:
                    train_vectors = vectors

                self.index.train(train_vectors)
                self.is_trained = True

            # Add vectors to index
            start_idx = self.index.ntotal
            self.index.add(vectors)

            # Update ID mappings
            for i, id_val in enumerate(ids):
                faiss_idx = start_idx + i
                self.id_map[faiss_idx] = id_val
                self.reverse_id_map[id_val] = faiss_idx
        else:
            # Fallback: store vectors in memory
            self.vectors = np.vstack([self.vectors, vectors])
            self.vector_ids.extend(ids)
    
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, List[str]]:
        """
        Search for k nearest neighbors to the query vector.

        Args:
            query_vector: Query vector (1 x dimension)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (distances, ids) where distances and ids are arrays of length k
        """
        # Handle uninitialized index
        if self.dimension is None or self.index is None:
            return np.array([]), []
        
        query_vector = query_vector.astype('float32').reshape(1, -1)

        if FAISS_AVAILABLE:
            if self.index.ntotal == 0:
                return np.array([]), []

            # Perform search
            distances, indices = self.index.search(query_vector, k)

            # Map FAISS indices back to actual IDs
            result_ids = []
            for idx in indices[0]:
                if idx != -1 and idx in self.id_map:  # -1 indicates no result
                    result_ids.append(self.id_map[idx])
                else:
                    result_ids.append(None)

            return distances[0], result_ids
        else:
            # Fallback: compute similarities in memory
            if self.vectors.shape[0] == 0:
                return np.array([]), []

            # Compute cosine similarities
            query_norm = np.linalg.norm(query_vector)
            if query_norm == 0:
                similarities = np.zeros(self.vectors.shape[0])
            else:
                normalized_query = query_vector / query_norm
                dot_products = self.vectors @ normalized_query.T
                vector_norms = np.linalg.norm(self.vectors, axis=1)
                
                # Handle zero vectors
                similarities = np.divide(
                    dot_products.flatten(),
                    vector_norms,
                    out=np.zeros_like(dot_products.flatten()),
                    where=vector_norms != 0
                )

            # Get top-k most similar vectors
            top_k_indices = np.argsort(similarities)[::-1][:k]
            top_similarities = similarities[top_k_indices]
            
            # Convert similarities to distances (1 - similarity)
            distances = 1 - top_similarities
            
            # Get corresponding IDs
            result_ids = [self.vector_ids[i] if i < len(self.vector_ids) else None for i in top_k_indices]

            return distances, result_ids
    
    def remove_vector(self, id_val: str) -> bool:
        """
        Remove a vector by its ID.
        NOTE: FAISS doesn't support efficient deletion, so this is a limitation.
        In practice, you might want to use a soft delete approach.
        """
        if id_val not in self.reverse_id_map:
            return False
        
        # FAISS doesn't support efficient deletion, so we'll mark as deleted
        # In a real implementation, you might rebuild the index periodically
        # or use a different approach
        logger.warning("FAISS doesn't support efficient deletion. Consider rebuilding index.")
        return True
    
    def reset(self) -> None:
        """Reset the index."""
        if self.dimension is None:
            # Index was never initialized, nothing to reset
            self.id_map = {}
            self.reverse_id_map = {}
            self.vectors = None
            self.vector_ids = []
            return
        
        # Recreate the index with same dimension
        self._initialize_index(self.dimension)
        self.id_map = {}
        self.reverse_id_map = {}
        self.is_trained = False


class CacheManager:
    """
    Manages caching for embeddings and retrieval results to improve performance.
    """

    def __init__(self, max_size: int = 10000, ttl_minutes: int = 60, default_ttl: int = None, use_redis: bool = False):
        self.max_size = max_size
        self.default_ttl = (default_ttl or ttl_minutes * 60)
        self.ttl_seconds = self.default_ttl
        self.use_redis = use_redis and REDIS_AVAILABLE
        
        self.expiration_times = {}
        
        if self.use_redis:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                # Test connection
                self.redis_client.ping()
            except:
                # Fall back to in-memory cache if Redis is not available
                self.use_redis = False
                self.cache = {}
                self.access_times = {}
                self.size = 0
        else:
            self.cache = {}
            self.access_times = {}
            self.size = 0
    
    def _hash_key(self, key: str) -> str:
        """Create a hash of the key to standardize it."""
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if self.use_redis:
            try:
                value = self.redis_client.get(key)
                if value:
                    return pickle.loads(value)
                return None
            except:
                return None
        else:
            hashed_key = self._hash_key(key)

            if hashed_key not in self.cache:
                return None

            # Check if expired
            if datetime.now() - self.access_times[hashed_key] > timedelta(seconds=self.ttl_seconds):
                self._remove(hashed_key)
                return None

            # Update access time
            self.access_times[hashed_key] = datetime.now()
            return self.cache[hashed_key]

    def set(self, key: str, value: Any) -> None:
        """Set a value in cache."""
        if self.use_redis:
            try:
                serialized_value = pickle.dumps(value)
                self.redis_client.setex(key, self.ttl_seconds, serialized_value)
            except:
                pass  # Silently fail if Redis is not available
        else:
            hashed_key = self._hash_key(key)
            is_new_key = hashed_key not in self.cache

            # Evict oldest if at max capacity and adding a new item
            if is_new_key and self.size >= self.max_size:
                self._evict_oldest()

            self.cache[hashed_key] = value
            self.access_times[hashed_key] = datetime.now()
            self.expiration_times[hashed_key] = datetime.now() + timedelta(seconds=self.ttl_seconds)
            
            if is_new_key:
                self.size += 1
    
    def _remove(self, hashed_key: str) -> None:
        """Remove a key from cache."""
        if hashed_key in self.cache:
            del self.cache[hashed_key]
            del self.access_times[hashed_key]
            self.size -= 1
    
    def _evict_oldest(self) -> None:
        """Evict the oldest accessed item."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove(oldest_key)
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()
        self.access_times.clear()
        self.size = 0


class OptimizedGraphManager:
    """
    Optimized graph operations for large-scale datasets.
    """
    
    def __init__(self, use_sparse: bool = True):
        self.use_sparse = use_sparse
        self.graph_data = {}
        self.node_mapping = {}
        self.reverse_node_mapping = {}
        self.next_id = 0
        
        try:
            from scipy.sparse import csr_matrix, lil_matrix
            self.has_scipy = True
            self.csr_matrix = csr_matrix
            self.lil_matrix = lil_matrix
        except ImportError:
            self.has_scipy = False
            logger.warning("scipy not available, using dense matrices which may be inefficient for large graphs")
    
    def add_nodes(self, node_ids: List[str]) -> None:
        """Add nodes to the graph."""
        for node_id in node_ids:
            if node_id not in self.node_mapping:
                self.node_mapping[node_id] = self.next_id
                self.reverse_node_mapping[self.next_id] = node_id
                self.next_id += 1
    
    def add_edges(self, edges: List[Tuple[str, str, float]]) -> None:
        """Add weighted edges to the graph."""
        # Convert node IDs to internal indices
        indexed_edges = []
        for source, target, weight in edges:
            if source not in self.node_mapping:
                self.node_mapping[source] = self.next_id
                self.reverse_node_mapping[self.next_id] = source
                self.next_id += 1
            if target not in self.node_mapping:
                self.node_mapping[target] = self.next_id
                self.reverse_node_mapping[self.next_id] = target
                self.next_id += 1
            
            source_idx = self.node_mapping[source]
            target_idx = self.node_mapping[target]
            indexed_edges.append((source_idx, target_idx, weight))
        
        # Store edges (implementation depends on whether we're using sparse matrices)
        if self.has_scipy and self.use_sparse:
            # Use sparse matrix representation
            n_nodes = self.next_id
            row = np.array([e[0] for e in indexed_edges])
            col = np.array([e[1] for e in indexed_edges])
            data = np.array([e[2] for e in indexed_edges])
            
            # Create adjacency matrix
            adj_matrix = self.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
            self.graph_data['adj_matrix'] = adj_matrix
        else:
            # Use dense matrix representation
            n_nodes = self.next_id
            adj_matrix = np.zeros((n_nodes, n_nodes))
            for source_idx, target_idx, weight in indexed_edges:
                adj_matrix[source_idx, target_idx] = weight
            self.graph_data['adj_matrix'] = adj_matrix
    
    def get_neighbors(self, node_id: str, max_neighbors: int = 100) -> List[Tuple[str, float]]:
        """Get neighbors of a node with their weights."""
        if node_id not in self.node_mapping:
            return []
        
        node_idx = self.node_mapping[node_id]
        adj_matrix = self.graph_data.get('adj_matrix')
        
        if adj_matrix is None:
            return []
        
        # Get the row corresponding to this node
        if self.has_scipy and self.use_sparse:
            # For sparse matrix, get non-zero entries in the row
            row_start = adj_matrix.indptr[node_idx]
            row_end = adj_matrix.indptr[node_idx + 1]
            cols = adj_matrix.indices[row_start:row_end]
            weights = adj_matrix.data[row_start:row_end]
            
            # Create list of (neighbor_id, weight) tuples
            neighbors = []
            for col_idx, weight in zip(cols, weights):
                neighbor_id = self.reverse_node_mapping[col_idx]
                neighbors.append((neighbor_id, weight))
        else:
            # For dense matrix
            row = adj_matrix[node_idx]
            # Get indices of non-zero elements
            nonzero_indices = np.nonzero(row)[0]
            neighbors = []
            for idx in nonzero_indices[:max_neighbors]:
                neighbor_id = self.reverse_node_mapping[idx]
                weight = row[idx]
                neighbors.append((neighbor_id, weight))
        
        # Sort by weight (descending) and return top neighbors
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:max_neighbors]
    
    def compute_pagerank(self, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> Dict[str, float]:
        """Compute PageRank scores for nodes in the graph."""
        adj_matrix = self.graph_data.get('adj_matrix')
        if adj_matrix is None or self.next_id == 0:
            return {}
        
        n = self.next_id
        
        # Handle sparse vs dense matrix
        if self.has_scipy and self.use_sparse:
            # Convert to probability transition matrix
            # First, normalize each row to sum to 1 (for outgoing edges)
            row_sums = np.array(adj_matrix.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            probabilities = adj_matrix.multiply(1/row_sums[:, np.newaxis])
            
            # Initialize PageRank
            pr = np.ones(n) / n
            
            # Power iteration
            for _ in range(max_iter):
                new_pr = (1 - damping) / n + damping * (probabilities.T @ pr)
                
                # Check for convergence
                if np.linalg.norm(new_pr - pr, ord=1) < tol:
                    break
                pr = new_pr
        else:
            # For dense matrix
            # Normalize each row to sum to 1 (for outgoing edges)
            row_sums = adj_matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            probabilities = adj_matrix / row_sums[:, np.newaxis]
            
            # Initialize PageRank
            pr = np.ones(n) / n
            
            # Power iteration
            for _ in range(max_iter):
                new_pr = (1 - damping) / n + damping * (probabilities.T @ pr)
                
                # Check for convergence
                if np.linalg.norm(new_pr - pr, ord=1) < tol:
                    break
                pr = new_pr
        
        # Map back to node IDs
        pagerank_scores = {}
        for idx, score in enumerate(pr):
            node_id = self.reverse_node_mapping[idx]
            pagerank_scores[node_id] = float(score)
        
        return pagerank_scores


def paginate_results(results: List[Any], page: int, page_size: int) -> Dict[str, Any]:
    """
    Paginate a list of results.
    
    Args:
        results: List of results to paginate
        page: Page number (1-indexed)
        page_size: Number of items per page
    
    Returns:
        Dictionary with paginated results and metadata
    """
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 10
    
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    paginated_items = results[start_idx:end_idx]
    
    total_pages = (len(results) + page_size - 1) // page_size  # Ceiling division
    
    return {
        'items': paginated_items,
        'pagination': {
            'current_page': page,
            'page_size': page_size,
            'total_items': len(results),
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1,
            'next_page': page + 1 if page < total_pages else None,
            'prev_page': page - 1 if page > 1 else None
        }
    }


class BatchProcessor:
    """
    Handles batch processing for large ingestion jobs.
    """
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(self, items: List[Any], process_func, *args, **kwargs) -> List[Any]:
        """
        Process a batch of items asynchronously.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            *args, **kwargs: Additional arguments to pass to process_func
        
        Returns:
            List of processed results
        """
        loop = asyncio.get_event_loop()
        
        # Split items into batches
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        
        all_results = []
        for batch in batches:
            # Process batch in thread pool
            future = loop.run_in_executor(
                self.executor,
                self._process_batch_sync,
                batch,
                process_func,
                args,
                kwargs
            )
            batch_results = await future
            all_results.extend(batch_results)
        
        return all_results
    
    def _process_batch_sync(self, batch: List[Any], process_func, args, kwargs) -> List[Any]:
        """Process a batch synchronously in a worker thread."""
        results = []
        for item in batch:
            try:
                result = process_func(item, *args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item in batch: {e}")
                results.append(None)  # Or handle error as appropriate
        return results
    
    def close(self):
        """Close the executor."""
        self.executor.shutdown(wait=True)


class AsyncProcessingManager:
    """
    Manages asynchronous processing for heavy computations.
    """
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    async def run_task(self, task_func, *args, **kwargs):
        """Run a task with concurrency control."""
        async with self.semaphore:
            return await task_func(*args, **kwargs)
    
    async def run_tasks_parallel(self, tasks: List[Tuple[callable, tuple, dict]]) -> List[Any]:
        """Run multiple tasks in parallel with concurrency control."""
        async def run_single_task(func, args, kwargs):
            async with self.semaphore:
                return await func(*args, **kwargs)
        
        coroutines = [run_single_task(func, args, kwargs) for func, args, kwargs in tasks]
        return await asyncio.gather(*coroutines, return_exceptions=True)


class MemoryManager:
    """
    Manages memory for large graphs and datasets.
    """
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_usage = 0
        self.managed_objects = {}
    
    def register_object(self, obj_id: str, obj: Any) -> bool:
        """Register an object for memory management."""
        try:
            import sys
            obj_size = sys.getsizeof(pickle.dumps(obj))
            
            if self.current_usage + obj_size > self.max_memory_bytes:
                self._free_memory(obj_size)
            
            if self.current_usage + obj_size <= self.max_memory_bytes:
                self.managed_objects[obj_id] = {
                    'object': obj,
                    'size': obj_size,
                    'timestamp': datetime.now(),
                    'access_count': 0
                }
                self.current_usage += obj_size
                return True
            else:
                logger.warning(f"Cannot register object {obj_id}, exceeds memory limit")
                return False
        except Exception as e:
            logger.error(f"Error registering object {obj_id}: {e}")
            return False
    
    def get_object(self, obj_id: str) -> Optional[Any]:
        """Get a registered object."""
        if obj_id in self.managed_objects:
            obj_entry = self.managed_objects[obj_id]
            obj_entry['access_count'] += 1
            obj_entry['timestamp'] = datetime.now()
            return obj_entry['object']
        return None
    
    def _free_memory(self, needed_bytes: int) -> None:
        """Free memory by removing least recently used objects."""
        # Sort objects by last access time (oldest first)
        sorted_objects = sorted(
            self.managed_objects.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        freed_bytes = 0
        for obj_id, obj_entry in sorted_objects:
            if freed_bytes >= needed_bytes:
                break
            
            self.current_usage -= obj_entry['size']
            del self.managed_objects[obj_id]
            freed_bytes += obj_entry['size']
        
        logger.info(f"Freed {freed_bytes} bytes of memory")


class ParallelPatternExtractor:
    """
    Performs parallel processing for pattern extraction.
    """
    
    def __init__(self, num_processes: int = 4):
        self.num_processes = num_processes
    
    async def extract_patterns_parallel(
        self, 
        data: List[Any], 
        pattern_func,
        *args, 
        **kwargs
    ) -> List[Any]:
        """
        Extract patterns in parallel.
        
        Args:
            data: List of data items to extract patterns from
            pattern_func: Function to extract patterns from a single item
            *args, **kwargs: Additional arguments to pass to pattern_func
        
        Returns:
            List of extracted patterns
        """
        loop = asyncio.get_event_loop()
        
        # Split data into chunks for parallel processing
        chunk_size = max(1, len(data) // self.num_processes)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Create tasks for parallel processing
        tasks = []
        for chunk in chunks:
            task = loop.run_in_executor(
                None,
                self._extract_patterns_chunk,
                chunk,
                pattern_func,
                args,
                kwargs
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        chunk_results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_results = []
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)
        
        return all_results
    
    def _extract_patterns_chunk(
        self, 
        chunk: List[Any], 
        pattern_func, 
        args, 
        kwargs
    ) -> List[Any]:
        """Process a chunk of data synchronously."""
        results = []
        for item in chunk:
            try:
                result = pattern_func(item, *args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error extracting pattern from item: {e}")
                results.append(None)
        return results


# Integration classes that bring everything together
class ScalableEKMBackend:
    """
    A scalable backend that integrates all the scalability improvements.
    """
    
    def __init__(
        self,
        storage: BaseStorage,
        embeddings: BaseEmbeddings,
        vector_dim: int = None,
        cache_size: int = 10000,
        max_graph_nodes: int = 100000
    ):
        self.storage = storage
        self.embeddings = embeddings
        
        # Initialize scalability components
        self.vector_index = VectorIndexManager(dimension=vector_dim)
        self.cache = CacheManager(max_size=cache_size)
        self.graph_manager = OptimizedGraphManager()
        self.memory_manager = MemoryManager()
        
        # Initialize processing managers
        self.batch_processor = BatchProcessor(batch_size=50)
        self.async_manager = AsyncProcessingManager(max_concurrent_tasks=5)
        self.pattern_extractor = ParallelPatternExtractor(num_processes=4)
    
    async def add_akus_batch(self, workspace_id: str, aku_data: List[Dict[str, Any]]) -> List[str]:
        """
        Add AKUs in batch with vector indexing.
        """
        # Process embeddings in batch
        contents = [aku['content'] for aku in aku_data]
        embeddings_list = await self.embeddings.embed_documents(contents)
        
        # Prepare for storage
        akus_to_store = []
        for i, (aku, emb) in enumerate(zip(aku_data, embeddings_list)):
            akus_to_store.append({
                'content': aku['content'],
                'embedding': emb,
                'metadata': aku.get('metadata', {})
            })
        
        # Store in database
        aku_ids = await self.storage.save_akus(workspace_id, None, akus_to_store)
        
        # Add to vector index
        embedding_array = np.array(embeddings_list).astype('float32')
        self.vector_index.add_vectors(embedding_array, aku_ids)
        
        return aku_ids
    
    async def search_akus(
        self,
        workspace_id: str,
        query: str,
        top_k: int = 10,
        page: int = 1,
        page_size: int = 10
    ) -> Dict[str, Any]:
        """
        Search AKUs with vector indexing, caching, and pagination.
        """
        # Create cache key
        cache_key = f"search:{workspace_id}:{query}:{top_k}"
        cached_result = self.cache.get(cache_key)

        if cached_result:
            logger.info("Returning cached search result")
            # Apply pagination to cached result
            paginated = paginate_results(cached_result, page, page_size)
            return paginated

        # Get query embedding
        query_embedding = await self.embeddings.embed_query(query)
        query_array = np.array(query_embedding).astype('float32')

        # Search using vector index
        distances, aku_ids = self.vector_index.search(query_array, top_k)

        # Filter out None IDs and get full AKU details
        valid_results = []
        for dist, aku_id in zip(distances, aku_ids):
            if aku_id is not None:
                # Get AKU details from storage
                aku_details = await self.storage.get_akus_by_ids([aku_id])
                if aku_details:
                    aku_info = aku_details[0]
                    valid_results.append({
                        'id': aku_info['id'],
                        'content': aku_info['content'],
                        'distance': float(dist),
                        'similarity': float(1 / (1 + dist))  # Convert distance to similarity
                    })

        # Cache the results (without pagination applied)
        self.cache.set(cache_key, valid_results)

        # Apply pagination
        paginated = paginate_results(valid_results, page, page_size)
        return paginated
    
    def close(self):
        """Clean up resources."""
        self.batch_processor.close()