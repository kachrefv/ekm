import logging
import uuid
import os
from typing import Optional, Dict

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from ekm.core.mesh import EKM
from ekm.storage.sql import SQLStorage
from ekm.providers.gemini import GeminiProvider
from ekm.core.models import Base, Setting
from .config import settings

logger = logging.getLogger(__name__)

# --- Database Setup ---
engine = create_async_engine(settings.DB_URL)
# Note: create_all doesn't work with async engines directly; tables should be created separately
# For development, you can run: `python -c "from ekm.core.models import Base; from sqlalchemy import create_engine; engine = create_engine('sqlite:///ekm.db'); Base.metadata.create_all(engine)"`
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession)

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        await db.close()

# --- EKM Singleton ---
_ekm_instance: Optional[EKM] = None
_ekm_config_cache: Dict = {}

async def get_ekm_instance(workspace_id: Optional[str] = None):
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
            await db_session.close()

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
                # Use async query
                from sqlalchemy import select
                result = await db_session.execute(
                    select(Setting).where(Setting.workspace_id == ws_uuid)
                )
                saved_settings = result.scalars().all()

                for s in saved_settings:
                    current_config[s.key] = s.value

                # Update the EKM instance's config
                _ekm_instance.config = current_config

            except Exception as e:
                logger.warning(f"Error loading settings for workspace {workspace_id}: {e}")
        finally:
            await db_session.close()

    return _ekm_instance
