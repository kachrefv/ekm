"""
EKM Command Line Interface - Tools for training and interactive chat.
"""
import asyncio
import argparse
import os
import sys
import uuid
from typing import List
from ekm.core.mesh import EKM
from ekm.core.agent import EKMAgent
from ekm.storage.sql import SQLStorage
from ekm.providers.gemini import GeminiProvider
from ekm.providers.base import BaseLLM, BaseEmbeddings
from ekm.utils.document_loader import DocumentLoader

# Configuration (In a real app, these would come from env vars or config file)
os.environ.setdefault("GEMINI_API_KEY", "")

async def get_or_create_workspace(storage: SQLStorage, name: str) -> str:
    """Resolve a human-readable name to a UUID string."""
    try:
        # Check if it's already a valid UUID
        import uuid
        uuid.UUID(name)
        return name
    except ValueError:
        # Look up by name or create
        from ekm.core.models import Workspace
        workspace = storage.db.query(Workspace).filter(Workspace.name == name).first()
        if not workspace:
            print(f" [+] Creating new workspace: {name}")
            workspace = Workspace(id=uuid.uuid4(), name=name, user_id="cli_user")
            storage.db.add(workspace)
            storage.db.commit()
        return str(workspace.id)

async def train_files(ekm: EKM, workspace_id: str, file_paths: List[str]):
    """Ingest content from files into the EKM."""
    print(f"\n--- Training EKM on {len(file_paths)} files ---")
    loader = DocumentLoader(llm_provider=ekm.llm)
    
    for path in file_paths:
        if not os.path.exists(path):
            print(f" [!] File not found: {path}")
            continue
            
        print(f" [+] Processing {os.path.basename(path)}...")
        try:
            content = await loader.load(path)
            await ekm.train(workspace_id, content)
            print(f" [✓] Ingested {len(content)} characters.")
        except Exception as e:
            print(f" [!] Error processing {path}: {e}")

async def chat_loop(ekm: EKM, workspace_id: str):
    """Start an interactive chat loop with the EKMAgent."""
    agent = EKMAgent(ekm, workspace_id)
    print(f"\n{'='*60}")
    print(" EKM INTERACTIVE AGENT (Type 'exit' or 'quit' to stop)")
    print(f" Workspace: {workspace_id}")
    
    # Check for consolidation
    from ekm.core.models import GKU
    gku_count = ekm.storage.db.query(GKU).filter(GKU.workspace_id == uuid.UUID(workspace_id)).count()
    if gku_count == 0:
        print(f" [!] NOTE: Mesh is not yet consolidated. Retrieval may be slower and less 'causal'.")
        print(f"     Run 'python ekm_cli.py sleep --workspace {workspace_id}' to optimize.")
    
    print(f"{'='*60}\n")
    
    while True:
        try:
            user_input = input("You> ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
                
            if not user_input:
                continue
                
            print("Thinking...", end="\r")
            result = await agent.chat(user_input)
            
            if "error" in result:
                print(f"Agent Error: {result['error']}")
                continue
                
            print(f"Agent ({result['mode_used']})> {result['response']}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[!] Error: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Episodic Knowledge Mesh CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train EKM on text, PDF, or image files")
    train_parser.add_argument("files", nargs="+", help="Paths to .txt, .md, .pdf, .jpg, .png files")
    train_parser.add_argument("--workspace", default=str(uuid.uuid4()), help="Workspace ID to use")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument("--workspace", required=True, help="Workspace ID to chat with")
    
    # Consolidation command
    sleep_parser = subparsers.add_parser("sleep", help="Run consolidation cycle")
    sleep_parser.add_argument("--workspace", required=True, help="Workspace ID")

    args = parser.parse_args()
    
    # Initialize EKM
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from ekm.core.models import Base
    
    db_url = "sqlite:///ekm.db"
    engine = create_engine(db_url)
    
    # Ensure tables exist
    Base.metadata.create_all(engine)
    
    # Optimized config for CLI usage
    config = {
        "EKM_SEMANTIC_THRESHOLD": 0.70,  # Lowered from 0.82 for better recall
        "VECTOR_DIMENSION": 3072        # Optimized for modern embedding models
    }
    
    SessionLocal = sessionmaker(bind=engine)
    db_session = SessionLocal()
    
    storage = SQLStorage(db=db_session)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        print(" [!] Please set GEMINI_API_KEY environment variable.")
        return

    provider = GeminiProvider(api_key=api_key)
    ekm = EKM(storage=storage, llm=provider, embeddings=provider, config=config)
    
    # Resolve workspace ID if present
    workspace_id = None
    if hasattr(args, 'workspace') and args.workspace:
        workspace_id = await get_or_create_workspace(storage, args.workspace)

    if args.command == "train":
        await train_files(ekm, workspace_id, args.files)
        print(f"\nDone. Use workspace ID: {args.workspace} (resolved to {workspace_id}) for chatting.")
        
    elif args.command == "chat":
        await chat_loop(ekm, workspace_id)
        
    elif args.command == "sleep":
        from ekm.core.consolidation import SleepConsolidator
        consolidator = SleepConsolidator(storage, provider, provider)
        print(f"Running sleep cycle for {workspace_id}...")
        results = await consolidator.run_consolidation(workspace_id)
        print(f"Consolidation complete: {results}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
