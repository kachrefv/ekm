# Configuration Guide

This guide explains how to configure the Episodic Knowledge Mesh (EKM) for different environments and use cases.

## Environment Variables

EKM uses environment variables for configuration. Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

### Database Configuration

- `DATABASE_URL`: Connection string for the database (default: `sqlite:///./ekm.db`)

### LLM Provider Configuration

Choose one of the following providers:

- `LLM_PROVIDER`: Provider to use (`openai`, `anthropic`, `cohere`, `groq`, `ollama`)
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `COHERE_API_KEY`: Cohere API key
- `GROQ_API_KEY`: Groq API key
- `OLLAMA_BASE_URL`: Ollama server URL (default: `http://localhost:11434`)

### Embedding Provider Configuration

- `EMBEDDING_PROVIDER`: Provider for embeddings (same options as LLM)
- `EMBEDDING_MODEL`: Specific embedding model to use

### Performance Settings

- `SEMANTIC_THRESHOLD`: Minimum similarity threshold (default: `0.82`)
- `TOP_K_RESULTS`: Number of results to return (default: `5`)
- `CHUNK_MIN_SIZE`: Minimum chunk size for text processing (default: `512`)
- `CHUNK_MAX_SIZE`: Maximum chunk size for text processing (default: `2048`)

### Scalability Settings

- `FAISS_ENABLED`: Enable FAISS for vector search (default: `true`)
- `REDIS_ENABLED`: Enable Redis caching (default: `false`)
- `REDIS_HOST`: Redis host (default: `localhost`)
- `REDIS_PORT`: Redis port (default: `6379`)

### Debug Settings

- `DEBUG_MODE`: Enable debug output (default: `false`)

## Programmatic Configuration

You can also configure EKM programmatically:

```python
from ekm import EKM
from ekm.storage.sql_storage import SQLStorage
from ekm.providers.openai import OpenAIProvider
from ekm.config.settings import Settings

# Create custom settings
settings = Settings()
settings.database_url = "postgresql://user:pass@localhost/db"
settings.semantic_threshold = 0.75

# Initialize components with custom settings
storage = SQLStorage(settings.database_url)
llm = OpenAIProvider(api_key="your-key")
embeddings = OpenAIProvider(api_key="your-key")

# Pass configuration to EKM
ekm = EKM(storage, llm, embeddings, config={
    "EKM_SEMANTIC_THRESHOLD": settings.semantic_threshold,
    "USE_ENHANCED_EMBEDDINGS": True,
    "BASE_EMBEDDING_MODEL": "all-MiniLM-L6-v2"
})
```

## Provider-Specific Configuration

### OpenAI

```python
from ekm.providers.openai import OpenAIProvider

llm = OpenAIProvider(
    api_key="your-api-key",
    model="gpt-4",  # or "gpt-3.5-turbo"
    temperature=0.7
)
```

### Anthropic

```python
from ekm.providers.anthropic import AnthropicProvider

llm = AnthropicProvider(
    api_key="your-api-key",
    model="claude-3-opus-20240229",
    temperature=0.7
)
```

### Ollama

```python
from ekm.providers.ollama import OllamaProvider

llm = OllamaProvider(
    base_url="http://localhost:11434",
    model="llama2",
    temperature=0.7
)
```

## Performance Tuning

### Memory Management

For large knowledge bases, configure memory management:

```python
config = {
    "CACHE_SIZE": 50000,  # Increase cache size
    "MAX_GRAPH_NODES": 500000,  # Increase graph node limit
    "VECTOR_DIMENSION": 1536  # Set based on your embedding model
}
ekm = EKM(storage, llm, embeddings, config=config)
```

### Batch Processing

Configure batch sizes for training:

```python
config = {
    "EKM_MIN_CHUNK_SIZE": 256,  # Smaller chunks for detailed knowledge
    "EKM_MAX_CHUNK_SIZE": 1024,  # Larger chunks for context
    "BATCH_SIZE": 10  # Process documents in batches
}
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Verify your API keys are set correctly in environment variables
2. **Database Connection**: Check that your database URL is correct and the database is accessible
3. **Performance Issues**: Adjust semantic thresholds and cache sizes based on your use case
4. **Memory Issues**: Reduce cache sizes or process smaller batches

### Debugging

Enable debug mode to get more detailed logs:

```bash
DEBUG_MODE=true python your_script.py
```

Or set in your configuration:

```python
config = {"DEBUG_MODE": True}
```