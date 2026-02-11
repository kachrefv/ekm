# Consolidation System Documentation

## Overview

The Consolidation System is a core component of the Episodic Knowledge Mesh (EKM) that implements a sleep-inspired four-phase process for reducing knowledge redundancy and forming higher-level abstractions. This system mimics biological memory consolidation during sleep, helping to maintain an efficient and organized knowledge base.

## Architecture

The consolidation system is implemented in the `SleepConsolidator` class located in `ekm/core/consolidation.py`. The system follows a modular design with each phase of the consolidation process handled by dedicated methods:

- `_replay_phase()`: Builds similarity graph of AKUs
- `_consolidate_phase()`: Detects and merges similar clusters
- `_reorganize_phase()`: Updates pattern tensors and GKU memberships
- `_prune_phase()`: Archives original AKUs and cleans up

## Four-Phase Process

### Phase 1: REPLAY
The REPLAY phase builds a similarity graph of recent Atomic Knowledge Units (AKUs). It compares embeddings using cosine similarity and creates connections between similar AKUs based on a configurable threshold.

**Key Features:**
- Uses FAISS for approximate nearest neighbor search on large datasets (>100 AKUs)
- Falls back to brute-force computation for smaller datasets
- Configurable similarity threshold (default: 0.85)

### Phase 2: CONSOLIDATE
The CONSOLIDATE phase detects connected components in the similarity graph and merges similar AKUs into comprehensive units. This phase identifies clusters of related knowledge and combines them.

**Key Features:**
- Finds connected components using BFS algorithm
- Merges clusters that meet minimum support requirements (default: 3 AKUs)
- Uses LLM to create comprehensive, conflict-resolved content
- Generates merged embeddings by averaging individual embeddings

### Phase 3: REORGANIZE
The REORGANIZE phase updates pattern tensors and forms new Global Knowledge Units (GKUs) from the remaining AKUs. This phase creates higher-level abstractions from the consolidated knowledge.

**Key Features:**
- Leverages advanced clustering algorithms from `ekm/core/clustering.py`
- Creates attention-weighted centroids for GKUs
- Generates enhanced pattern signatures with graph-based features
- Computes quality metrics for clustering results

### Phase 4: PRUNE
The PRUNE phase archives the original AKUs that were consolidated, cleaning up the knowledge base and maintaining efficiency.

**Key Features:**
- Marks original AKUs as archived in batch operations
- Maintains references to original AKUs in metadata
- Preserves provenance information

## Configuration Parameters

The `SleepConsolidator` accepts the following configuration parameters:

- `replay_threshold`: Minimum similarity threshold for connecting AKUs (default: 0.85)
- `consolidate_threshold`: Threshold for consolidation decisions (default: 0.90)
- `min_cluster_support`: Minimum number of AKUs required to form a cluster (default: 3)

## Security Features

The consolidation system includes several security measures:

- **Input Sanitization**: All AKU content is sanitized before LLM calls to prevent prompt injection attacks
- **Length Limiting**: Input content is limited to 10,000 characters to prevent resource exhaustion
- **Content Filtering**: Removes potential prompt injection markers from input

## Performance Optimizations

The system includes several performance optimizations:

- **Approximate Nearest Neighbor Search**: Uses FAISS library for efficient similarity computation on large datasets
- **Batch Processing**: Performs database operations in batches where possible
- **Memory Efficiency**: Normalizes embeddings to reduce memory footprint during computation
- **Fallback Mechanisms**: Gracefully degrades to alternative algorithms when primary methods fail

## Quality Metrics

The consolidation process calculates several quality metrics:

- **Compression Ratio**: Measures reduction in AKU count
- **Semantic Preservation**: Estimates how much meaning was preserved during consolidation
- **Clustering Quality**: Various metrics including silhouette score, Calinski-Harabasz score, and Davies-Bouldin score

## Usage Example

```python
from ekm.core.consolidation import SleepConsolidator
from ekm.storage.sqlite_storage import SQLiteStorage  # or your preferred storage

# Initialize components
storage = SQLiteStorage(db_path="ekm.db")
llm = YourLLMProvider()  # e.g., OpenAI, Anthropic, etc.
embeddings = YourEmbeddingProvider()  # e.g., OpenAI, SentenceTransformer, etc.

# Create consolidator with custom parameters
consolidator = SleepConsolidator(
    storage=storage,
    llm=llm,
    embeddings=embeddings,
    replay_threshold=0.8,
    min_cluster_support=2
)

# Run consolidation on a workspace
result = await consolidator.run_consolidation(workspace_id="your-workspace-id")

print(f"Compression ratio: {result.compression_ratio:.2%}")
print(f"Semantic preservation: {result.semantic_preservation:.2%}")
print(f"New AKUs created: {len(result.consolidated_akus)}")
print(f"AKUs archived: {len(result.archived_akus)}")
print(f"GKUs created: {len(result.created_gkus)}")
```

## Best Practices

1. **Regular Consolidation**: Run consolidation periodically to maintain knowledge base efficiency
2. **Monitor Metrics**: Track compression ratio and semantic preservation to ensure quality
3. **Adjust Thresholds**: Tune similarity thresholds based on your domain and requirements
4. **Backup Before Consolidation**: Consider backing up the knowledge base before major consolidation runs
5. **Resource Management**: Consolidation can be resource-intensive; schedule during low-usage periods

## Troubleshooting

### Common Issues

- **High Memory Usage**: Large knowledge bases may require more memory during consolidation. Consider increasing available RAM or running consolidation in batches.
- **Slow Performance**: If FAISS is not available, the system falls back to O(n²) similarity computation which can be slow for large datasets.
- **Poor Quality Results**: Adjust similarity thresholds and minimum cluster support based on your knowledge domain.

### Performance Tips

- Install FAISS for optimal performance: `pip install faiss-cpu` or `pip install faiss-gpu`
- Use appropriate embedding dimensions for your use case
- Monitor LLM token usage to manage costs
- Consider running consolidation asynchronously for large knowledge bases

## Future Enhancements

Planned improvements to the consolidation system include:

- Incremental consolidation for continuous knowledge base maintenance
- Parallel processing for faster computation
- Advanced semantic validation of consolidated content
- Improved rollback mechanisms for failed consolidations
- Better integration with active learning systems