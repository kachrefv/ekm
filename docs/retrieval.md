# Retrieval System in Episodic Knowledge Mesh (EKM)

## Overview

The retrieval system in the Episodic Knowledge Mesh (EKM) is a sophisticated multi-layered architecture designed to efficiently retrieve relevant knowledge from large-scale knowledge graphs. It combines multiple retrieval strategies including attention mechanisms, hierarchical search, and reinforcement learning to provide contextually relevant results.

## Architecture

### Multi-Modal Retrieval

The EKM retrieval system supports three distinct retrieval modes:

#### 1. Hybrid Mode (Default)
- Combines all knowledge layers (episodic, AKU, GKU)
- Provides comprehensive results across all knowledge types
- Balances breadth and depth of retrieval

#### 2. Episodic Mode
- Focuses exclusively on episodic memory layer
- Retrieves personal experiences and temporal sequences
- Ideal for context-dependent queries

#### 3. Facts Mode (`mode='facts'`)
- Retrieves only atomic facts (AKUs): those in the best-matching GKUs plus a direct search over all facts
- Skips the episodic layer (raw source chunks)
- Best for conceptual questions answered from extracted facts
- Formerly called "causal" mode. `mode='causal'` still works but is deprecated: it never followed cause-and-effect links (relationships have a `causal_weight` field, but nothing sets it yet)

### How a query is answered

1. **Embed the query once.** The same embedding is reused by every layer and by the RL reranker.
2. **Start the query-graph extraction** (an LLM call feeding the structural head) in the background, so it overlaps with the steps below. It is skipped when the structural head weight is 0.
3. **Episodic layer:** episodes of the workspace with cosine similarity of at least 0.3 (`EPISODIC_MIN_SIMILARITY`).
4. **Facts layer, stage 1 (GKU selection):** every GKU is scored with the three heads below, combined as a weighted average (default 0.6 semantic / 0.25 structural / 0.15 temporal). GKUs that contain facts from the focus buffer get a bounded boost (at most +0.15). The best `top_k` GKUs are kept (at least 5 with `use_hierarchical_attention`).
5. **Facts layer, stage 2:** all active facts of those GKUs are ranked by cosine similarity, and a direct similarity search over all facts of the workspace adds facts whose GKU was not selected (or that have no GKU yet, before the first consolidation).
6. **One score scale.** Every result's `score` is its cosine similarity to the query. A fact inside a selected GKU gets a bonus of up to `GKU_CONTEXT_BOOST` (0.1), in proportion to the GKU's score. The bonus only adds, so being in a cluster never lowers a fact. Episodes and facts are therefore comparable when merged.
7. **Optional RL reranking** (keeps this order until the RL model has learned from feedback).
8. **Diversity:** at most 2 results from the same source episode (the chunk itself plus facts extracted from it) before other sources get a turn; deferred items still fill the list if nothing else is left.

Similarity is cosine on every storage backend (pgvector, SQLite and in-memory), and episode search on SQLite is scoped per workspace.

### Attention heads

All three heads return raw scores (no softmax: a softmax over values in [-1, 1] is nearly uniform and erased the differences between GKUs).

#### Semantic head
Cosine similarity between the query embedding and each GKU's centroid.

#### Structural head
Similarity between the query's entity-relation pattern (extracted by the LLM) and each GKU's pattern signature, via `calculate_pattern_similarity`.

#### Temporal head
`exp(-|now - t| / 30 days)`, where `t` is the median time of the GKU's facts (or its creation time). Since queries are always "now", this is a recency preference.

### What is not used during retrieval

`SparseAttention`, `MultiHeadAttention` and the hierarchical coarse/fine attention helpers exist in `ekm/core/attention/mechanisms.py`, but retrieval does not call them, and `use_sparse_attention` is accepted only for API compatibility. Despite its name, `QKVAttentionRetriever` has no learned Q/K/V projections: each head is a similarity function.

### 3. Reinforcement Learning Integration

The retrieval system incorporates RL feedback for continuous improvement:

#### Feature Engineering
The RL system uses 10 different features for each query-response pair:
1. Semantic similarity
2. Focus buffer boost
3. Recency score
4. Frequency score
5. Confidence score
6. Diversity score
7. Length score
8. Complexity score
9. Relevance score
10. Contextual score

#### Model Updates
```python
async def _update_model(self, feedback_record: FeedbackRecord):
    # Extract features for the feedback record
    features = await self._extract_features(rl_state, response)
    
    # Calculate predicted vs actual scores
    predicted_score = np.dot(features, self.weights)
    actual_score = feedback_record.user_rating
    error = actual_score - predicted_score
    
    # Update weights using gradient descent
    self.weights += self.learning_rate * error * features
    self.weights = np.clip(self.weights, -10, 10)  # Prevent explosion
```

### 4. Focus Buffer Integration

The system incorporates context awareness through focus buffers:

```python
def compute_focus_boosts(focus_buffer, gku_ids):
    focus_boosts = np.zeros(len(gku_ids))
    if focus_buffer and hasattr(focus_buffer, 'items'):
        for i, gku_id in enumerate(gku_ids):
            # Check if AKUs in this GKU are in focus
            for item_id, item in focus_buffer.items.items():
                if item_id in gku_related_items:
                    focus_boosts[i] += item.current_weight
    return focus_boosts
```

- Context-aware retrieval based on current focus
- Boosts relevance of knowledge units related to current context
- Maintains conversational coherence

## Scalability Features

### 1. Vector Indexing with FAISS
- Efficient similarity search using FAISS indexing
- Multiple index types (Flat, IVF, HNSW) for different use cases
- Fallback to in-memory operations when FAISS unavailable

### 2. Caching Mechanisms
- Embedding caching to avoid recomputation
- Result caching for frequent queries
- TTL-based cache expiration
- Redis support for distributed caching

### 3. Memory Management
- Monitors memory usage for large graphs
- Automatic eviction of least recently used objects
- Size estimation for stored objects

### 4. Batch Processing
- Asynchronous batch processing for large ingestion jobs
- Configurable batch sizes and worker counts
- Thread pool execution for CPU-bound operations

## Performance Characteristics

### Time Complexity
- **Standard attention**: O(N²) - typically avoided
- **Sparse attention**: O(N*k) where k is average connections per node
- **Hierarchical attention**: O(N₁ + N₂) where N₁, N₂ are intermediate layer sizes
- **Graph operations**: O(E + V) for sparse graphs where E is edges, V is vertices

### Space Complexity
- **Attention weights**: O(N*k) for sparse, O(N²) for dense
- **Pattern signatures**: O(C) where C is number of clusters
- **Cached embeddings**: O(M*D) where M is cache size, D is embedding dimension
- **Graph storage**: O(E) for sparse adjacency lists

## Integration Points

### 1. Storage Layer
- Interfaces with BaseStorage for GKU/AKU retrieval
- Supports various storage backends
- Handles embedding and metadata retrieval

### 2. Embedding Providers
- Compatible with various embedding services
- Supports batch embedding operations
- Handles embedding normalization

### 3. LLM Integration
- Works with different LLM providers
- Supports query understanding and expansion
- Enables semantic interpretation

## Configuration Options

### Retrieval Parameters
- `top_k`: Number of results to return
- `mode`: `'hybrid'` (default), `'facts'` or `'episodic'`
- `focus_buffer`: Facts the conversation is already about; their GKUs get a bounded boost
- `use_adaptive_weights`: Read head weights from the service's shared weighting (changed only through `RetrievalService.record_head_feedback`)
- `use_hierarchical_attention`: Select at least 5 GKUs instead of exactly `top_k`
- `use_rl_feedback`: Enable RL-based reranking
- `debug_mode`: Add per-head debug info (`debug_info`) and a ranking explanation (`metadata['explanation']`)
- `use_sparse_attention`: Ignored; kept for API compatibility

### Tuning constants (`ekm/core/retrieval/service.py`)
- `GKU_CONTEXT_BOOST` (0.1): Largest bonus a GKU can add to a fact's score
- `EPISODIC_MIN_SIMILARITY` (0.3): Cosine cut-off for episodes
- `MAX_RESULTS_PER_SOURCE` (2): Results per source episode before others get a turn
- `DEFAULT_HEAD_WEIGHTS` (`ekm/core/attention/mechanisms.py`): Weights of the semantic, structural and temporal heads

Measure changes with `benchmarks/retrieval_quality.py`, which reports recall@k and MRR per configuration (for example with and without the structural head) on labelled queries.

## Quality Assurance

### Evaluation Metrics
- **Precision/Recall**: Traditional IR metrics
- **Diversity**: Coverage of different knowledge aspects
- **Coherence**: Logical consistency of results
- **Freshness**: Temporal relevance of results

### Monitoring
- Performance metrics tracking
- Quality score monitoring
- User satisfaction measurement
- System health indicators

## Best Practices

### For Developers
1. Use appropriate retrieval mode based on query type
2. Monitor memory usage with large knowledge graphs
3. Tune hyperparameters based on domain characteristics
4. Implement proper error handling and fallbacks

### For Users
1. Provide feedback to improve RL system
2. Use specific queries for better precision
3. Consider temporal aspects when relevant
4. Understand the trade-offs between speed and accuracy

## Future Enhancements

### Planned Improvements
- Cross-modal attention for multi-type knowledge
- Automated hyperparameter tuning
- Federated learning for privacy-preserving improvements
- Attention mechanism transfer learning
- Advanced query understanding

The EKM retrieval system represents a sophisticated approach to knowledge retrieval that balances efficiency, relevance, and adaptability while maintaining scalability for large knowledge graphs.