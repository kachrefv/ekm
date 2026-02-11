import logging
import uuid
import os
from typing import Optional, Dict

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ekm.core.mesh import EKM
from ekm.storage.sql import SQLStorage
from ekm.providers.gemini import GeminiProvider
from ekm.core.models import Base, Setting
from .config import settings

logger = logging.getLogger(__name__)

# --- Database Setup ---
engine = create_engine(settings.DB_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- EKM Singleton ---
_ekm_instance: Optional[EKM] = None
_ekm_config_cache: Dict = {}

def get_ekm_instance(workspace_id: Optional[str] = None):
    """
    Returns a Singleton instance of EKM.
    Notes:
    - EKM itself is stateful regarding storage/LLM, but config might change per workspace.
    - If workspace_id is provided, we update the config of the singleton.
    """
    global _ekm_instance
    
    # 1. Initialize if not exists
    if _ekm_instance is None:
        logger.info("Initializing EKM Singleton...")
        db_session = SessionLocal()
        try:
            storage = SQLStorage(db=db_session)
            # Use environment variable or fallback
            api_key = settings.GEMINI_API_KEY
            provider = GeminiProvider(api_key=api_key)
            
            _ekm_instance = EKM(
                storage=storage, 
                llm=provider, 
                embeddings=provider, 
                config=settings.DEFAULT_EKM_CONFIG.copy()
            )
        finally:
            db_session.close()

    # 2. Update config for workspace if needed
    if workspace_id:
        # Check if we need to reload settings for this workspace
        # For simplicity in this singleton, we just query and update.
        # A more robust solution might handle concurrent requests for different workspaces better,
        # but EKM's config is mostly used at runtime.
        db_session = SessionLocal()
        try:
            try:
                ws_uuid = uuid.UUID(workspace_id) if isinstance(workspace_id, str) else workspace_id
                
                # Merge default config with saved settings
                current_config = settings.DEFAULT_EKM_CONFIG.copy()
                saved_settings = db_session.query(Setting).filter(Setting.workspace_id == ws_uuid).all()
                
                for s in saved_settings:
                    current_config[s.key] = s.value
                
                # Update the EKM instance's config
                _ekm_instance.config = current_config
                
            except Exception as e:
                logger.warning(f"Error loading settings for workspace {workspace_id}: {e}")
        finally:
            db_session.close()
            
    return _ekm_instance
