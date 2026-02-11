# Troubleshooting Guide

This guide helps troubleshoot common issues with the Episodic Knowledge Mesh (EKM).

## Common Issues

### 1. API Key Errors

**Problem**: Getting authentication errors when using LLM/embedding providers.

**Solutions**:
- Verify your API key is correct and has proper permissions
- Check that your environment variables are set correctly:
  ```bash
  echo $OPENAI_API_KEY  # Should show your API key
  ```
- Ensure you're using the right provider in your configuration

### 2. Database Connection Issues

**Problem**: Cannot connect to the database.

**Solutions**:
- For SQLite: Ensure the file path exists and is writable
- For PostgreSQL: Verify connection string format and credentials
- Check that the database server is running
- Initialize the database if it's the first run:
  ```python
  from ekm.storage.sql_storage import SQLStorage
  import asyncio
  
  async def init_db():
      storage = SQLStorage("sqlite:///./ekm.db")
      await storage.init_db()
  
  asyncio.run(init_db())
  ```

### 3. Memory Issues

**Problem**: Running out of memory with large knowledge bases.

**Solutions**:
- Reduce cache sizes in configuration:
  ```python
  config = {"CACHE_SIZE": 1000}  # Smaller cache
  ```
- Process documents in smaller batches
- Use memory-efficient providers (like Ollama with smaller models)
- Increase system memory if possible

### 4. Slow Performance

**Problem**: Retrieval or training taking too long.

**Solutions**:
- Ensure FAISS is installed and enabled:
  ```bash
  pip install faiss-cpu  # or faiss-gpu
  ```
- Check that semantic thresholds are appropriate for your use case
- Consider using sparse attention mechanisms
- Optimize your hardware (GPU acceleration for embeddings)

### 5. No Results Returned

**Problem**: Queries returning no results.

**Solutions**:
- Lower the semantic threshold temporarily to see if results appear
- Verify that training was successful and knowledge was stored
- Check that the workspace ID matches between training and retrieval
- Ensure embeddings are being generated correctly

## Debugging Steps

### 1. Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set DEBUG_MODE in your configuration
config = {"DEBUG_MODE": True}
```

### 2. Check Component Initialization

```python
from ekm import EKM
from ekm.storage.sql_storage import SQLStorage
from ekm.providers.openai import OpenAIProvider

# Test each component separately
storage = SQLStorage("sqlite:///./ekm.db")
try:
    # Test storage connection
    await storage.get_workspaces()
    print("Storage connection OK")
except Exception as e:
    print(f"Storage error: {e}")

llm = OpenAIProvider(api_key="your-key")
try:
    # Test LLM connection
    result = await llm.generate("Test", "Hello")
    print("LLM connection OK")
except Exception as e:
    print(f"LLM error: {e}")
```

### 3. Verify Training Success

```python
# Check if training completed successfully
workspace_id = "test_workspace"
await ekm.train(workspace_id, "Some test content")

# Verify data was stored
akus = await storage.get_akus_by_workspace(workspace_id)
print(f"Stored {len(akus)} AKUs")
```

### 4. Test Retrieval Isolation

```python
# Test retrieval with a simple query
results = await ekm.retrieve(workspace_id, "test")
print(f"Retrieved {len(results['results'])} results")

# Check the content of results
for result in results['results'][:2]:  # Show first 2
    print(f"Content: {result['content'][:100]}...")
```

## Provider-Specific Issues

### OpenAI

- **Rate Limits**: Check your account's rate limits and billing status
- **Model Availability**: Verify the model you're requesting is available in your region

### Anthropic

- **Request Format**: Ensure your prompts comply with Claude's format requirements
- **Token Limits**: Claude has specific token limits for input and output

### Ollama

- **Server Running**: Ensure the Ollama server is running (`ollama serve`)
- **Model Downloaded**: Verify the requested model is downloaded (`ollama pull model_name`)
- **Hardware Acceleration**: Check that GPU drivers are properly configured if using GPU

## Performance Optimization

### 1. Profile Your Code

```python
import cProfile
import pstats

def profile_ekm_operation():
    # Your EKM operations here
    pass

# Profile the operation
pr = cProfile.Profile()
pr.enable()
profile_ekm_operation()
pr.disable()

# Print stats
stats = pstats.Stats(pr)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

### 2. Monitor Resource Usage

```python
import psutil
import os

def monitor_resources():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"CPU usage: {process.cpu_percent()}%")
```

## Getting Help

### 1. Check Logs

Look for detailed error messages in the logs, especially when running in debug mode.

### 2. Verify Dependencies

```bash
pip list | grep -E "(ekm|openai|anthropic|faiss|sqlalchemy)"
```

### 3. Create Minimal Reproduction

Create a minimal script that reproduces the issue:

```python
# minimal_reproduction.py
from ekm import EKM
# ... minimal code to reproduce the issue
```

### 4. Community Support

If you're still having issues:
- Check the GitHub issues page
- Create a new issue with your problem
- Include your environment details and reproduction steps