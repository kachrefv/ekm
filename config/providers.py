"""Provider configuration for EKM."""
import os
from typing import Dict, Any
from .settings import Settings


def get_provider_config(settings: Settings) -> Dict[str, Any]:
    """Get provider-specific configuration."""
    config = {
        'llm': {
            'provider': settings.llm_provider,
            'api_key': getattr(settings, f'{settings.llm_provider}_api_key', None),
            'model': os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
        },
        'embedding': {
            'provider': settings.embedding_provider,
            'api_key': getattr(settings, f'{settings.embedding_provider}_api_key', None),
            'model': os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002'),
        }
    }
    
    # Add provider-specific settings
    if settings.llm_provider == 'ollama':
        config['llm']['base_url'] = settings.ollama_base_url
        
    if settings.embedding_provider == 'ollama':
        config['embedding']['base_url'] = settings.ollama_base_url
        
    return config