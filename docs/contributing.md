# Contributing to Episodic Knowledge Mesh

We welcome contributions to the Episodic Knowledge Mesh project! This document outlines the process for contributing.

## Getting Started

1. Fork the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev,test]"
   ```

## Development Workflow

1. Create a branch for your feature/fix:
   ```bash
   git checkout -b feature/my-feature
   ```
2. Make your changes
3. Add tests for your changes
4. Run tests:
   ```bash
   make test
   ```
5. Format your code:
   ```bash
   make format
   ```
6. Submit a pull request

## Code Structure

The project follows a modular architecture:

- `ekm/core/` - Core functionality (attention, retrieval, knowledge extraction)
- `ekm/providers/` - LLM and embedding provider integrations
- `ekm/storage/` - Storage implementations
- `tests/` - Test suite
- `docs/` - Documentation
- `scripts/` - Utility scripts

## Testing

All contributions must include appropriate tests:

- Unit tests for individual components
- Integration tests for workflows
- Performance benchmarks when applicable

Run all tests with:
```bash
make test
```

## Code Standards

- Follow PEP 8 guidelines
- Use type hints for all public functions
- Write docstrings for all public classes and functions
- Keep functions focused and reasonably sized

## Questions?

Feel free to open an issue if you have questions about contributing.