import os

from ekm.providers.openrouter import DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL


class Settings:
    DB_URL = os.environ.get("EKM_DB_URL", "sqlite+aiosqlite:///ekm.db")
    # Set via environment variable; never hardcode keys.
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    OPENROUTER_LLM_MODEL = os.environ.get("OPENROUTER_LLM_MODEL", DEFAULT_LLM_MODEL)
    OPENROUTER_EMBEDDING_MODEL = os.environ.get("OPENROUTER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    # VECTOR_DIMENSION is left unset: it is detected from the embedding model
    DEFAULT_EKM_CONFIG = {
        "EKM_SEMANTIC_THRESHOLD": 0.70,
        "EKM_MIN_CHUNK_SIZE": 512,
        "EKM_MAX_CHUNK_SIZE": 2048,
        "RL_LEARNING_RATE": 0.1
    }

settings = Settings()
