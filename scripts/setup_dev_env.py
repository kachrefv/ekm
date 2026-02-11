#!/usr/bin/env python3
"""Setup development environment for EKM."""

import os
import sys
import subprocess
from pathlib import Path


def setup_dev_environment():
    """Setup development environment."""
    print("Setting up development environment for EKM...")
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    
    # Setup database
    print("Initializing database...")
    from ekm.storage.sql_storage import SQLStorage
    import asyncio
    
    async def init_db():
        storage = SQLStorage("sqlite:///./ekm.db")
        await storage.init_db()
    
    asyncio.run(init_db())
    
    print("Development environment setup complete!")


if __name__ == "__main__":
    setup_dev_environment()