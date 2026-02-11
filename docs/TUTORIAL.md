# Episodic Knowledge Mesh (EKM) Comprehensive Tutorial

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Core Concepts](#core-concepts)
- [Advanced Features](#advanced-features)
- [Storage Backends](#storage-backends)
- [Sleep-Inspired Consolidation](#sleep-inspired-consolidation)
- [Attention Mechanisms](#attention-mechanisms)
- [Batch Operations](#batch-operations)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Introduction

The Episodic Knowledge Mesh (EKM) is a state-of-the-art knowledge management system that combines episodic memory preservation, sparse pattern tensor representations, and attention-based retrieval to achieve both scalability and semantic depth. This tutorial provides a comprehensive guide to using all features of the EKM library.

### What is EKM?

EKM addresses the fundamental tension between scalability and structural expressiveness in knowledge management systems. Traditional vector databases scale efficiently but lose relational context, while knowledge graphs maintain structure but struggle with scale. EKM bridges this gap through:

- **Sparse Pattern Tensors**: k-nearest neighbor graphs achieving O(N·k) complexity
- **Three-Layer Memory Hierarchy**: Episodic, Atomic Knowledge Units (AKUs), and Global Knowledge Units (GKUs)
- **QKV Attention-Based Retrieval**: Context-aware, explainable results
- **Reinforcement Learning Feedback**: Continuous improvement from user interactions
- **Focus Buffer Integration**: Dynamic prioritization of relevant knowledge

### Prerequisites

Before using EKM, ensure you have:
- Python 3.8 or higher
- pip package manager
- Basic understanding of embeddings and attention mechanisms

## Installation

### Basic Installation

Install EKM using pip:

```bash
pip install ekm
```

### Optional Dependencies

For enhanced functionality, install additional packages:

```bash
# For enhanced embeddings
pip install sentence-transformers

# For vector databases
pip install faiss-cpu pgvector

# For graph operations
pip install scipy networkx

# For visualization
pip install matplotlib seaborn plotly
```

## Getting Started

### Basic Setup

Begin by importing EKM and initializing a basic instance:

```python
from ekm import EKM, create_ekm_with_memory_storage
from ekm.providers.mock import MockLLM, MockEmbeddings

# Initialize EKM with memory storage (for development)
llm = MockLLM()
embeddings = MockEmbeddings()
ekm = create_ekm_with_memory_storage(llm, embeddings)

# Create a workspace for your knowledge
workspace_id = "my_first_ekm_workspace"
```

### Adding Knowledge

Train the EKM with your knowledge base:

```python
# Train the EKM with foundational knowledge
await ekm.train(workspace_id,
    "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data.",
    title="ML Definition"
)

await ekm.train(workspace_id,
    "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
    title="Deep Learning Basics"
)
```

### Retrieving Information

Retrieve relevant information using natural language queries:

```python
# Retrieve information
results = await ekm.retrieve(workspace_id, "What is machine learning?")

# Display results
for result in results['results']:
    print(f"Content: {result['content'][:100]}...")
    print(f"Score: {result['score']:.3f}")
    print(f"Layer: {result.get('layer', 'N/A')}")
    print("---")
```

## Core Concepts

### Memory Hierarchy

EKM implements a three-layer memory hierarchy:

#### Episodic Memory Layer
Preserves raw experiences with contextual summaries and vector embeddings.

#### Atomic Knowledge Units (AKUs)
Self-contained factual statements with dual embeddings and sparse pattern tensors.

#### Global Knowledge Units (GKUs)
Emergent concept clusters formed through pattern similarity analysis.

### Sparse Pattern Tensors

Knowledge relationships are represented as k-nearest neighbor graphs with edge attributes, achieving O(N·k) complexity versus O(N²) for dense approaches.

### QKV Attention-Based Retrieval

Uses multi-head attention to combine semantic, structural, and temporal retrieval heads, providing context-aware results with interpretable attention weights.

## Advanced Features

### Custom Configuration

Configure EKM with custom parameters:

```python
from ekm import create_ekm_with_memory_storage
from ekm.providers.mock import MockLLM, MockEmbeddings

# Configure EKM with custom parameters
config = {
    'EKM_MIN_CHUNK_SIZE': 500,          # Minimum text chunk size
    'EKM_MAX_CHUNK_SIZE': 2000,         # Maximum text chunk size
    'EKM_SEMANTIC_THRESHOLD': 0.8,      # Similarity threshold for grouping
    'VECTOR_DIMENSION': 1536,           # Embedding dimension (adjust for your model)
    'CACHE_SIZE': 5000,                 # Size of internal cache
    'MAX_GRAPH_NODES': 50000,           # Maximum nodes in knowledge graph
    'USE_ENHANCED_EMBEDDINGS': False,   # Whether to use enhanced embeddings
    'USE_SPARSE_ATTENTION': True,       # Enable sparse attention
    'USE_ADAPTIVE_WEIGHTS': True,       # Enable adaptive head weighting
    'USE_HIERARCHICAL_ATTENTION': True, # Enable hierarchical attention
    'RL_LEARNING_RATE': 0.1,            # Learning rate for RL system
    'RL_EXPLORATION_RATE': 0.1,         # Exploration rate for RL system
    'RL_MEMORY_SIZE': 1000              # Memory size for RL system
}

llm = MockLLM()
embeddings = MockEmbeddings()
ekm = create_ekm_with_memory_storage(llm, embeddings, config=config)
```

### Advanced Retrieval Options

Use specific parameters and attention mechanisms:

```python
# Retrieve with specific parameters and attention mechanisms
results = await ekm.retrieve(
    workspace_id,
    query="How does deep learning work?",
    top_k=5,                           # Number of results to return
    use_sparse_attention=True,         # Use sparse attention for efficiency
    use_adaptive_weights=True,         # Use adaptive head weighting
    use_hierarchical_attention=True,   # Use two-stage hierarchical attention
    layers=['episodic', 'aku', 'gku']  # Search across all layers
)

# Examine detailed metadata
print(f"Retrieved {len(results['results'])} results")
print(f"Metadata: {results['metadata']}")
print(f"Layer distribution: {results['metadata']['layer_distribution']}")
```

### Focus Buffer for Targeted Retrieval

Use focus buffers to prioritize certain knowledge units:

```python
from ekm.core.state import FocusBuffer

# Create a focus buffer to prioritize certain knowledge units
focus_buffer = FocusBuffer()
focus_buffer.update(["important_topic_id"], action='activate')

# Use focus buffer in retrieval to boost relevant results
results = await ekm.retrieve(
    workspace_id,
    query="Tell me about important topics",
    focus_buffer=focus_buffer
)
```

### Reinforcement Learning Feedback System

The EKM includes a reinforcement learning system that learns from user interactions:

```python
# Retrieve with RL feedback enabled (default behavior)
results = await ekm.retrieve(
    workspace_id,
    query="What is quantum computing?",
    use_rl_feedback=True,  # Enable RL-based reranking
    top_k=5
)

# Record user feedback to improve future responses
await ekm.record_response_feedback(
    query="What is quantum computing?",
    response_id=results['results'][0]['id'],
    response_content=results['results'][0]['content'],
    user_rating=0.8  # Rating from -1 (very bad) to 1 (very good)
)

# Record feedback for multiple responses at once
responses_with_ratings = [
    (results['results'][0]['id'], results['results'][0]['content'], 0.9),  # Good response
    (results['results'][1]['id'], results['results'][1]['content'], -0.3), # Poor response
    (results['results'][2]['id'], results['results'][2]['content'], 0.5)   # OK response
]
await ekm.batch_record_feedback(
    query="What is quantum computing?",
    responses_with_ratings=responses_with_ratings
)

# Check RL model performance
performance = await ekm.get_rl_model_performance()
print(f"Total feedback records: {performance['feedback_records']}")
print(f"Average rating: {performance['average_rating']:.3f}")
```

## Storage Backends

### Memory Storage (Development/Testing)

Best for development and testing:

```python
from ekm import create_ekm_with_memory_storage
from ekm.providers.mock import MockLLM, MockEmbeddings

# Initialize with memory storage
llm = MockLLM()
embeddings = MockEmbeddings()
ekm = create_ekm_with_memory_storage(llm, embeddings)
```

### SQL Storage (Production)

For production environments:

```python
from ekm import create_ekm_with_sql_storage
from sqlalchemy import create_engine
from ekm.providers.openai import OpenAILLM, OpenAIEmbeddings

# Setup database engine
engine = create_engine("postgresql://user:password@localhost/dbname")
session = engine.connect()

# Initialize with SQL storage
llm = OpenAILLM(api_key="your-api-key")
embeddings = OpenAIEmbeddings(api_key="your-api-key")
ekm = create_ekm_with_sql_storage(session, llm, embeddings)
```

## Sleep-Inspired Consolidation

Periodically consolidate knowledge to improve efficiency:

```python
# Perform consolidation to compress knowledge
consolidation_results = await ekm.consolidate(workspace_id)

print(f"Consolidated AKUs: {consolidation_results['consolidated_akus']}")
print(f"Created GKUs: {consolidation_results['created_gkus']}")
print(f"Compression Ratio: {consolidation_results['compression_ratio']}")
```

## Attention Mechanisms

### Sparse Attention

Use sparse attention for efficiency:

```python
# Use sparse attention mechanism
results = await ekm.retrieve(
    workspace_id,
    query="What are the key concepts?",
    use_sparse_attention=True,  # Enable sparse attention
    use_qkv_attention=True
)

# The results will include information about sparse attention usage
if 'metadata' in results and 'attention_interpretations' in results['metadata']:
    interpretations = results['metadata']['attention_interpretations']
    if 'use_sparse_attention' in interpretations:
        print(f"Sparse attention used: {interpretations['use_sparse_attention']}")
```

### Adaptive Head Weighting

Use adaptive head weighting to adjust attention based on query characteristics:

```python
# Enable adaptive head weighting
results = await ekm.retrieve(
    workspace_id,
    query="Explain the temporal aspects of the system",
    use_adaptive_weights=True  # Enable adaptive head weighting
)

# View the adaptive weights used
if 'metadata' in results and 'attention_interpretations' in results['metadata']:
    interpretations = results['metadata']['attention_interpretations']
    if 'head_weights_used' in interpretations:
        weights = interpretations['head_weights_used']
        print(f"Semantic weight: {weights['semantic']:.3f}")
        print(f"Structural weight: {weights['structural']:.3f}")
        print(f"Temporal weight: {weights['temporal']:.3f}")
```

### Hierarchical Attention

Use hierarchical attention for two-stage retrieval:

```python
# Use hierarchical attention for two-stage retrieval
results = await ekm.retrieve(
    workspace_id,
    query="What are the main topics covered?",
    use_hierarchical_attention=True,  # Enable hierarchical attention
    use_sparse_attention=True,
    use_adaptive_weights=True
)

# Hierarchical attention results include stage information
if 'metadata' in results and 'interpretations' in results:
    interpretations = results['interpretations']
    if 'hierarchical_process' in interpretations:
        print(f"Hierarchical process used: {interpretations['hierarchical_process']}")
        print(f"Stage 1 clusters selected: {interpretations['stage_1_clusters_selected']}")
        print(f"Stage 2 items selected: {interpretations['stage_2_items_selected']}")
```

## Batch Operations

Process multiple documents at once:

```python
# Process multiple documents at once
documents = [
    "Document 1 content...",
    "Document 2 content...",
    "Document 3 content..."
]

for doc in documents:
    await ekm.train(workspace_id, doc)

# Or process in batches
batch_size = 10
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    for doc in batch:
        await ekm.train(workspace_id, doc)
```

## API Reference

### EKM Class Methods

#### `train(workspace_id, text, title=None)`
Trains the EKM with new text data.

**Parameters:**
- `workspace_id` (str): ID of the workspace
- `text` (str): Text content to train on
- `title` (str, optional): Title for the training episode

**Returns:** None

#### `retrieve(workspace_id, query, top_k=5, focus_buffer=None, use_rl_feedback=True, use_sparse_attention=True, use_adaptive_weights=True, use_hierarchical_attention=False, layers=None)`
Retrieves relevant knowledge from the mesh.

**Parameters:**
- `workspace_id` (str): ID of the workspace to search in
- `query` (str): Query string
- `top_k` (int): Number of top results to return
- `focus_buffer` (FocusBuffer, optional): Focus buffer to influence retrieval
- `use_rl_feedback` (bool): Whether to use reinforcement learning feedback
- `use_sparse_attention` (bool): Whether to use sparse attention
- `use_adaptive_weights` (bool): Whether to use adaptive head weighting
- `use_hierarchical_attention` (bool): Whether to use hierarchical attention
- `layers` (List[str]): Layers to search in ('episodic', 'aku', 'gku')

**Returns:** Dict with results and metadata

#### `consolidate(workspace_id)`
Performs sleep-inspired consolidation to compress knowledge.

**Parameters:**
- `workspace_id` (str): ID of the workspace to consolidate

**Returns:** Dict with consolidation results

#### `record_response_feedback(query, response_id, response_content, user_rating, context=None)`
Records user feedback for a specific query-response pair.

**Parameters:**
- `query` (str): The original query
- `response_id` (str): ID of the response
- `response_content` (str): Content of the response
- `user_rating` (float): Rating from -1 (very bad) to 1 (very good)
- `context` (Dict[str, Any], optional): Additional context information

**Returns:** True if feedback was recorded successfully

#### `get_rl_model_performance()`
Gets performance metrics for the RL feedback system.

**Returns:** Dictionary with performance metrics

#### `batch_record_feedback(query, responses_with_ratings)`
Records feedback for multiple responses to the same query.

**Parameters:**
- `query` (str): The original query
- `responses_with_ratings` (List[Tuple[str, str, float]]): List of (response_id, content, rating) tuples

**Returns:** True if all feedback was recorded successfully

## Best Practices

### Knowledge Organization
- Organize knowledge into logical chunks of 500-2000 tokens
- Use descriptive titles for training episodes
- Regularly consolidate knowledge to maintain efficiency

### Query Optimization
- Use specific, clear queries for better results
- Leverage focus buffers for targeted retrieval
- Monitor and provide feedback to improve results over time

### Performance Tuning
- Adjust chunk sizes based on your content type
- Tune semantic thresholds for your domain
- Use appropriate storage backends for your scale

## Troubleshooting

### Common Issues

#### Slow Performance
- Check if you're using sparse attention
- Verify your storage backend is optimized
- Consider consolidating knowledge regularly

#### Poor Retrieval Quality
- Adjust semantic thresholds
- Provide feedback to the RL system
- Ensure your training data is well-structured

## Conclusion

The Episodic Knowledge Mesh provides a powerful platform for managing complex knowledge systems with both scalability and semantic depth. By leveraging sparse pattern tensors, attention mechanisms, and reinforcement learning, EKM offers a unique approach to knowledge management that adapts and improves over time.

This tutorial covered all major features of the EKM library, from basic setup to advanced configurations. For more information, refer to the API documentation and example projects in the repository.