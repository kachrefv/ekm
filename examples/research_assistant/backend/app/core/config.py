import os

class Settings:
    DB_URL = os.environ.get("EKM_DB_URL", "sqlite:///ekm.db")
    # SECURITY NOTE: In production, this must be set via environment variable. 
    # Fallback is only for local dev convenience if user hasn't set it yet.
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 
    
    DEFAULT_EKM_CONFIG = {
        "EKM_SEMANTIC_THRESHOLD": 0.70,
        "VECTOR_DIMENSION": 3072,
        "EKM_MIN_CHUNK_SIZE": 512,
        "EKM_MAX_CHUNK_SIZE": 2048,
        "RL_LEARNING_RATE": 0.1
    }

settings = Settings()
