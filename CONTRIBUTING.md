# Contributing to Episodic Knowledge Mesh (EKM)

Thank you for your interest in contributing to the Episodic Knowledge Mesh (EKM) project! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Fork and Clone the Repository
1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/YOUR_USERNAME/episodic-knowledge-mesh.git
cd episodic-knowledge-mesh
```

### Setting Up Development Environment
1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
   - On Windows:
   ```bash
   .venv\Scripts\activate
   ```
   - On Unix/Mac:
   ```bash
   source .venv/bin/activate
   ```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Development Environment

### Optional Dependencies
For enhanced functionality during development, you may want to install additional packages:

```bash
# For enhanced embeddings
pip install sentence-transformers

# For vector databases
pip install faiss-cpu pgvector

# For graph operations
pip install scipy networkx

# For visualization
pip install matplotlib seaborn plotly

# For development tools
pip install black flake8 mypy pytest pytest-asyncio
```

### Project Structure
```
episodic-knowledge-mesh/
├── ekm/                    # Main source code
│   ├── core/               # Core EKM functionality
│   │   ├── mesh.py         # Main EKM implementation
│   │   ├── attention.py    # Attention mechanisms
│   │   ├── clustering.py   # Clustering algorithms
│   │   ├── consolidation.py # Consolidation algorithms
│   │   ├── graph.py        # Graph management
│   │   ├── patterns.py     # Pattern tensor implementation
│   │   ├── state.py        # State management (FocusBuffer)
│   │   └── ...
│   ├── storage/            # Storage implementations
│   │   ├── base.py         # Base storage interface
│   │   ├── memory.py       # In-memory storage
│   │   ├── sql.py          # SQL storage
│   │   └── factory.py      # Storage factory
│   ├── providers/          # LLM and embedding providers
│   │   ├── base.py         # Base provider interface
│   │   ├── mock.py         # Mock provider for testing
│   │   ├── openai.py       # OpenAI provider
│   │   ├── gemini.py       # Google Gemini provider
│   │   └── deepseek.py     # DeepSeek provider
│   └── agents/             # Agent implementations
│       └── chat.py         # Chat agent
├── tests/                  # Test files
├── examples/               # Example usage
├── docs/                   # Documentation
├── pyproject.toml          # Project configuration
├── README.md               # Main documentation
├── LICENSE                 # License information
└── CONTRIBUTING.md         # This file
```

## How to Contribute

### Reporting Bugs
- Use the GitHub issue tracker
- Provide a clear title and description
- Include steps to reproduce the issue
- Specify your environment (OS, Python version, etc.)
- Include relevant error messages and stack traces

### Suggesting Features
- Use the GitHub issue tracker
- Describe the feature and its use case
- Explain why it would be beneficial
- Consider implementation complexity

### Code Contributions
1. Find an issue to work on or propose a new feature
2. Fork the repository
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Add tests if applicable
6. Update documentation if needed
7. Submit a pull request

## Coding Standards

### Python Style
- Follow PEP 8 style guide
- Use type hints for all function parameters and return values
- Write comprehensive docstrings for all public functions and classes
- Use descriptive variable and function names
- Keep functions focused and reasonably sized

### Example of Good Code Style
```python
from typing import List, Dict, Any, Optional
import numpy as np

def calculate_similarity(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vector1: First vector for comparison
        vector2: Second vector for comparison
        
    Returns:
        Cosine similarity value between -1 and 1
    """
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return float(dot_product / (norm_v1 * norm_v2))
```

### Async/Await Patterns
- Use async/await consistently for I/O operations
- Follow proper exception handling in async functions
- Use appropriate async libraries for concurrency

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=ekm

# Run specific test file
pytest tests/test_ekm.py

# Run tests with verbose output
pytest -v
```

### Writing Tests
- Write unit tests for new functionality
- Use pytest for test organization
- Follow AAA pattern (Arrange, Act, Assert)
- Test edge cases and error conditions
- Keep tests focused and fast

### Example Test
```python
import pytest
import asyncio
from ekm.core.mesh import EKM
from ekm.providers.mock import MockLLM, MockEmbeddings

@pytest.mark.asyncio
async def test_ekm_train_and_retrieve():
    # Arrange
    llm = MockLLM()
    embeddings = MockEmbeddings()
    ekm = EKM(storage=None, llm=llm, embeddings=embeddings)
    
    # Act
    await ekm.train("test_workspace", "Test knowledge content")
    results = await ekm.retrieve("test_workspace", "Test query")
    
    # Assert
    assert len(results['results']) >= 0
```

## Documentation

### Docstrings
- Use Google-style docstrings
- Document all public methods and classes
- Include type hints in docstrings for complex types
- Provide examples where helpful

### Example Docstring
```python
def retrieve(
    self, 
    workspace_id: str, 
    query: str, 
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Retrieve relevant knowledge from the mesh using attention mechanisms.
    
    Args:
        workspace_id: ID of the workspace to search in
        query: Query string to search for
        top_k: Number of top results to return (default: 5)
        
    Returns:
        Dictionary containing results and metadata
        
    Example:
        >>> results = await ekm.retrieve("workspace1", "What is AI?", top_k=3)
        >>> print(results['results'][0]['content'])
        "Artificial Intelligence is..."
    """
```

### README Updates
- Update README when adding major features
- Include usage examples
- Update installation instructions if needed

## Pull Request Process

### Before Submitting
1. Ensure all tests pass
2. Add new tests for new functionality
3. Update documentation as needed
4. Follow coding standards
5. Write clear, descriptive commit messages

### Creating a Pull Request
1. Push your feature branch to your fork
2. Go to the original repository on GitHub
3. Click "New pull request"
4. Select your branch
5. Fill out the pull request template:
   - Describe the changes
   - Explain the problem being solved
   - Include any relevant issue numbers
   - Note any breaking changes

### Review Process
- Maintainers will review your code
- Address any feedback promptly
- Be prepared to make changes
- Once approved, your PR will be merged

### Commit Message Guidelines
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## Community

### Getting Help
- Check the documentation first
- Search existing issues
- Ask questions in new issues if needed
- Be patient with responses

### Communication
- Be respectful and constructive
- Provide helpful feedback
- Welcome newcomers
- Share knowledge generously

## Questions?

If you have questions about contributing, feel free to open an issue or reach out to the maintainers.

Thank you for contributing to the Episodic Knowledge Mesh project!