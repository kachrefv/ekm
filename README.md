# Episodic Knowledge Mesh (EKM)
![Version](https://img.shields.io/badge/version-0.1.9-blue)

The Episodic Knowledge Mesh is an advanced knowledge management system that implements sophisticated attention mechanisms and retrieval-augmented generation (RAG) techniques. It organizes knowledge into Atomic Knowledge Units (AKUs) and Global Knowledge Units (GKUs) with rich relationship graphs.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Examples](#examples)
- [Configuration](#configuration)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Episodic Knowledge Mesh (EKM) is a cutting-edge knowledge management system that revolutionizes how information is stored, organized, and retrieved. By leveraging advanced attention mechanisms inspired by transformer architectures, EKM creates a dynamic mesh of interconnected knowledge units that can be efficiently queried and expanded.

### Core Concepts
- **Atomic Knowledge Units (AKUs)**: Fundamental building blocks of knowledge, each containing a single, coherent piece of information
- **Global Knowledge Units (GKUs)**: Conceptual clusters that group related AKUs together
- **Attention Mechanisms**: Multi-dimensional relevance scoring combining semantic, structural, and temporal factors
- **Hierarchical Retrieval**: Two-stage process for efficient knowledge discovery

## Key Features

### 1. Advanced Attention Mechanisms
- **Multi-Head Attention**: Combines semantic, structural, and temporal relevance
- **Sparse Attention**: Reduces computational complexity from O(N²) to O(N*k)
- **Hierarchical Attention**: Two-stage retrieval (GKU selection → AKU selection)
- **Adaptive Weighting**: Dynamically adjusts head importance based on query characteristics

### 2. Knowledge Organization
- **Atomic Knowledge Units (AKUs)**: Fine-grained knowledge atoms for precise retrieval
- **Global Knowledge Units (GKUs)**: Conceptual clusters for semantic grouping
- **Rich Relationships**: Graph-based connections between knowledge units
- **Pattern Signatures**: Structural characterization using graphlets and degree statistics

### 3. Training System
- **Single-Step Extraction**: Simultaneous extraction of summaries and AKUs
- **Semantic Chunking**: Intelligent text segmentation preserving meaning
- **Relationship Generation**: Automatic connection discovery using vector similarity
- **Batch Processing**: Efficient ingestion of large document collections

### 4. Retrieval System
- **Multi-Modal Search**: Hybrid, episodic, and causal retrieval modes
- **Context Awareness**: Focus buffer integration for conversational understanding
- **RL Feedback**: Continuous improvement through user interaction learning
- **Explainability**: Detailed reasoning for retrieval decisions

### 5. Scalability Features
- **Vector Indexing**: FAISS integration for efficient similarity search
- **Caching**: Multi-tier caching with Redis support
- **Memory Management**: Automatic resource optimization
- **Batch Operations**: Parallel processing for large-scale operations

## Project Structure

```
episodic_knowledge_mesh/
├── ekm/                           # Core EKM implementation
│   ├── __init__.py                # Main package exports
│   ├── core/                      # Core functionality
│   │   ├── __init__.py
│   │   ├── attention/             # Attention mechanisms
│   │   │   ├── __init__.py
│   │   │   ├── mechanisms.py      # Attention implementations
│   │   │   └── sparse_attention.py # Sparse attention
│   │   ├── retrieval/             # Retrieval system
│   │   │   ├── __init__.py
│   │   │   ├── service.py         # Retrieval service
│   │   │   └── fusion.py          # Result fusion logic
│   │   ├── knowledge/             # Knowledge extraction
│   │   │   ├── __init__.py
│   │   │   ├── extraction.py      # Knowledge extraction
│   │   │   └── organization.py    # Knowledge organization
│   │   ├── graph/                 # Graph operations
│   │   │   ├── __init__.py
│   │   │   └── operations.py      # Graph operations
│   │   ├── agent.py               # EKM agent implementation
│   │   ├── clustering.py          # Clustering algorithms
│   │   ├── consolidation.py       # Knowledge consolidation
│   │   ├── explainability.py      # Explainability features
│   │   ├── factory.py             # Factory patterns
│   │   ├── mesh.py                # Main EKM orchestrator
│   │   ├── models.py              # Data models
│   │   ├── patterns.py            # Pattern matching
│   │   ├── prompts.py             # LLM prompts
│   │   ├── query_analysis.py      # Query analysis
│   │   ├── rl_feedback.py         # Reinforcement learning feedback
│   │   ├── scalability.py         # Scalability features
│   │   ├── state.py               # State management
│   │   ├── text_processor.py      # Text processing
│   │   ├── training.py            # Training service
│   │   ├── utils.py               # Core utilities
│   │   └── validation.py          # Validation utilities
│   ├── providers/                 # LLM/embedding providers
│   │   ├── __init__.py
│   │   ├── base.py                # Base provider classes
│   │   ├── factory.py             # Provider factory
│   │   ├── openai.py              # OpenAI provider
│   │   ├── anthropic.py           # Anthropic provider
│   │   ├── azure_openai.py        # Azure OpenAI provider
│   │   ├── cohere.py              # Cohere provider
│   │   ├── google_genai.py        # Google GenAI provider
│   │   ├── groq.py                # Groq provider
│   │   ├── ollama.py              # Ollama provider
│   │   └── transformers.py        # Transformers provider
│   └── storage/                   # Storage implementations
│       ├── __init__.py
│       ├── base.py                # Base storage interface
│       ├── factory.py             # Storage factory
│       ├── sql_storage.py         # SQL storage implementation
│       └── utils.py               # Storage utilities
├── config/                        # Configuration
│   ├── __init__.py
│   ├── settings.py                # Application settings
│   └── providers.py               # Provider configuration
├── data/                          # Data files
│   └── __init__.py
├── docs/                          # Documentation
│   ├── __init__.py
│   ├── architecture/              # Architecture docs
│   │   ├── decisions/             # Architecture Decision Records
│   │   ├── diagrams/              # Architecture diagrams
│   │   └── overview.md            # Architecture overview
│   ├── api/                       # API documentation
│   ├── guides/                    # User guides
│   │   ├── getting_started.md     # Getting started guide
│   │   ├── configuration.md       # Configuration guide
│   │   └── troubleshooting.md     # Troubleshooting guide
│   ├── attention.md               # Attention mechanisms documentation
│   ├── retrieval.md               # Retrieval system documentation
│   ├── training.md                # Training system documentation
│   └── contributing.md            # Contribution guide
├── examples/                      # Example implementations
│   └── research_assistant/        # Research assistant example
│       ├── server.py              # EKM server
│       └── frontend/              # Frontend implementation
├── migrations/                    # Database migrations
│   └── __init__.py
├── notebooks/                     # Jupyter notebooks
│   └── exploration.ipynb          # Knowledge exploration notebook
├── results/                       # Output and results
│   └── __init__.py
├── scripts/                       # Utility scripts
│   ├── __init__.py
│   ├── setup_dev_env.py           # Development setup
│   ├── run_tests.py               # Test runner
│   └── deploy.py                  # Deployment script
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── conftest.py                # Test configuration
│   ├── unit/                      # Unit tests
│   │   ├── __init__.py
│   │   └── core/                  # Core unit tests
│   │       ├── attention/         # Attention tests
│   │       ├── retrieval/         # Retrieval tests
│   │       └── knowledge/         # Knowledge tests
│   └── integration/               # Integration tests
│       ├── __init__.py
│       ├── api/                   # API tests
│       └── services/              # Service tests
├── benchmarks/                    # Performance benchmarks
│   └── performance.py             # Performance tests
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
├── backend.spec                   # Backend specification
├── CONTRIBUTING.md                # Contribution guidelines
├── docker-compose.yml             # Docker configuration
├── ekm_cli.py                     # Command-line interface
├── LICENSE                        # License information
├── Makefile                       # Make commands
├── pyproject.toml                 # Python project configuration
├── pytest.ini                     # Pytest configuration
├── README.md                      # This file
├── requirements.txt               # Python dependencies
└── test_persona_logic.py          # Persona logic tests
```

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- An API key for your chosen LLM provider (OpenAI, Anthropic, etc.)

### Quick Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/episodic-knowledge-mesh.git
   cd episodic-knowledge-mesh
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```bash
   pip install -e .
   ```

5. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

## Quick Start

### Basic Usage
```python
import asyncio
from ekm import EKM
from ekm.storage.sql_storage import SQLStorage
from ekm.providers.openai import OpenAIProvider

async def main():
    # Initialize components
    storage = SQLStorage("sqlite:///./ekm.db")
    llm = OpenAIProvider(api_key="your-api-key")
    embeddings = OpenAIProvider(api_key="your-api-key")

    # Create EKM instance
    ekm = EKM(storage, llm, embeddings)

    # Train with some text
    workspace_id = "my_workspace"
    sample_text = "Artificial intelligence is intelligence demonstrated by machines..."
    await ekm.train(workspace_id, sample_text, title="AI Introduction")

    # Retrieve information
    query = "What is artificial intelligence?"
    results = await ekm.retrieve(workspace_id, query)
    
    print(f"Found {len(results['results'])} results:")
    for result in results['results']:
        print(f"- {result['content'][:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

### Using the CLI
```bash
python ekm_cli.py --help
```

## Architecture

### Core Components
The EKM system consists of several interconnected components:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │    │   Core Services  │    │  Infrastructure │
│     Layer       │    │      Layer       │    │     Layer       │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│   EKM Client    │◄──►│  • Training     │◄──►│ • Storage       │
│   EKM Agent     │    │  • Retrieval    │    │ • LLM Provider  │
│   CLI Tools     │    │  • Attention    │    │ • Embeddings    │
│   API Layer     │    │  • RL Feedback  │    │ • Vector Index  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Attention Mechanism Architecture
The attention system implements a multi-head approach:

1. **Semantic Head**: Matches query embeddings with GKU content embeddings using cosine similarity
2. **Structural Head**: Matches query patterns with GKU structural signatures using graph-based features
3. **Temporal Head**: Considers time-based relevance with exponential decay

### Knowledge Organization
- **AKUs (Atomic Knowledge Units)**: Individual facts or concepts stored as embeddings
- **GKUs (Global Knowledge Units)**: Clusters of related AKUs with centroid embeddings
- **Relationship Graph**: Connections between AKUs based on semantic similarity and contextual relationships

## Examples

### Research Assistant
Check the `examples/research_assistant/` directory for a complete implementation showing how to use EKM in a research assistant application with both server and frontend components.

### Jupyter Notebook
Explore knowledge management concepts interactively with the notebook in `notebooks/exploration.ipynb`.

## Configuration

### Environment Variables
Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

### Programmatic Configuration
```python
from ekm import EKM
from ekm.storage.sql_storage import SQLStorage
from ekm.providers.openai import OpenAIProvider

# Initialize with custom configuration
ekm = EKM(storage, llm, embeddings, config={
    "EKM_SEMANTIC_THRESHOLD": 0.75,
    "USE_ENHANCED_EMBEDDINGS": True,
    "BASE_EMBEDDING_MODEL": "all-MiniLM-L6-v2"
})
```

See the `docs/guides/configuration.md` file for detailed configuration options.

## Development

### Running Tests
```bash
make test           # Run all tests
make test-unit      # Run unit tests
make test-integration  # Run integration tests
```

### Code Formatting
```bash
make format    # Format code
make lint      # Run linters
```

### Running Benchmarks
```bash
make benchmark # Run performance benchmarks
```

### Docker Setup
```bash
docker-compose up -d  # Start services
```

## Contributing

We welcome contributions to the Episodic Knowledge Mesh project! Please see `CONTRIBUTING.md` for guidelines on how to contribute.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for your changes
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the terms found in the `LICENSE` file.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.