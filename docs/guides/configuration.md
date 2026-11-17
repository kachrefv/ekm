# Configuration Guide

This guide explains how to configure the Episodic Knowledge Mesh (EKM) for different environments and use cases.

## Environment Variables

EKM uses environment variables for configuration. Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

### Database Configuration

- `DATABASE_URL`: Connection string for the database (default: `sqlite:///./ekm.db`)

### LLM and Embedding Providers (OpenRouter)

EKM uses [OpenRouter](https://openrouter.ai) for both the LLM and the embeddings, so one key covers both:

- `OPENROUTER_API_KEY`: OpenRouter API key (required)
- `OPENROUTER_LLM_MODEL`: LLM model ID (default: `anthropic/claude-haiku-5.5`)
- `OPENROUTER_EMBEDDING_MODEL`: Embedding model ID (default: `openai/text-embedding-3-small`, 1536 dimensions). The list is at `https://openrouter.ai/api/v1/embeddings/models`.

Changing the embedding model changes the vector space. Facts embedded with one model can't be compared with queries embedded by another, so re-train existing workspaces after switching.

```python
from ekm.providers.openrouter import OpenRouterProvider

provider = OpenRouterProvider()  # reads the variables above
ekm = EKM(storage, llm=provider, embeddings=provider)
```

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