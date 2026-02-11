# Getting Started Guide

This guide will help you get started with the Episodic Knowledge Mesh (EKM).

## Prerequisites

- Python 3.8+
- pip package manager
- An API key for your chosen LLM provider (OpenAI, Anthropic, etc.)

## Installation

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

3. Install the package:
   ```bash
   pip install -e .
   ```

4. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Quick Start

### 1. Initialize EKM

```python
import asyncio
from ekm.core.mesh import EKM
from ekm.storage.sql_storage import SQLStorage
from ekm.providers.openai import OpenAIProvider

# Initialize components
storage = SQLStorage("sqlite:///./ekm.db")
llm = OpenAIProvider(api_key="your-api-key")
embeddings = OpenAIProvider(api_key="your-api-key")

# Create EKM instance
ekm = EKM(storage, llm, embeddings)
```

### 2. Create a Workspace and Train

```python
# Create a workspace
workspace_id = "my_workspace"

# Sample text to train on
sample_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, 
in contrast to the natural intelligence displayed by humans and animals.
"""

# Train the EKM
await ekm.train(workspace_id, sample_text, title="AI Introduction")
```

### 3. Retrieve Information

```python
# Query the EKM
query = "What is artificial intelligence?"
results = await ekm.retrieve(workspace_id, query)

print(f"Found {len(results['results'])} results:")
for result in results['results']:
    print(f"- {result['content'][:100]}...")
```

## Configuration

See the [Configuration Guide](configuration.md) for detailed configuration options.