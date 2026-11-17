"""Provider configuration for EKM."""
import os
from typing import Dict, Any
from .settings import Settings


def get_provider_config(settings: Settings) -> Dict[str, Any]:
    """Get provider-specific configuration."""
    from ekm.providers.openrouter import DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL

    config = {
        'llm': {
            'provider': settings.llm_provider,
            'api_key': getattr(settings, f'{settings.llm_provider}_api_key', None),
            'model': os.getenv('OPENROUTER_LLM_MODEL') or os.getenv('LLM_MODEL', DEFAULT_LLM_MODEL),
        },
        'embedding': {
            'provider': settings.embedding_provider,
            'api_key': getattr(settings, f'{settings.embedding_provider}_api_key', None),
            'model': os.getenv('OPENROUTER_EMBEDDING_MODEL') or os.getenv('EMBEDDING_MODEL', DEFAULT_EMBEDDING_MODEL),
        }
    }
    
    # Add provider-specific settings
    if settings.llm_provider == 'ollama':
        config['llm']['base_url'] = settings.ollama_base_url
        
    if settings.embedding_provider == 'ollama':
        config['embedding']['base_url'] = settings.ollama_base_url
        
    return config