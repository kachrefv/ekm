"""Configuration settings for EKM."""
import os
from typing import Optional


class Settings:
    """Application settings."""
    
    def __init__(self):
        self.database_url: str = os.getenv("DATABASE_URL", "sqlite:///./ekm.db")
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
        self.embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
        self.cohere_api_key: Optional[str] = os.getenv("COHERE_API_KEY")
        self.groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
        self.ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Performance settings
        self.semantic_threshold: float = float(os.getenv("SEMANTIC_THRESHOLD", "0.82"))
        self.top_k_results: int = int(os.getenv("TOP_K_RESULTS", "5"))
        self.chunk_min_size: int = int(os.getenv("CHUNK_MIN_SIZE", "512"))
        self.chunk_max_size: int = int(os.getenv("CHUNK_MAX_SIZE", "2048"))
        
        # Scalability settings
        self.faiss_enabled: bool = os.getenv("FAISS_ENABLED", "true").lower() == "true"
        self.redis_enabled: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
        self.redis_host: str = os.getenv("REDIS_HOST", "localhost")
        self.redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
        
        # Debug settings
        self.debug_mode: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"