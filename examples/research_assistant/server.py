import os
import uuid
import json
import asyncio
import logging
import numpy as np
from typing import List, Optional, Any, Dict

logger = logging.getLogger(__name__)
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Custom JSON encoder to handle numpy types
import fastapi.encoders as encoders

# Monkey patch the jsonable_encoder to handle numpy types
original_jsonable_encoder = encoders.jsonable_encoder

def patched_jsonable_encoder(obj, **kwargs):
    from numpy import float32, float64, int32, int64, ndarray
    import numpy as np
    
    if isinstance(obj, (float32, float64)):
        return float(obj)
    elif isinstance(obj, (int32, int64)):
        return int(obj)
    elif isinstance(obj, ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    
    return original_jsonable_encoder(obj, **kwargs)

encoders.jsonable_encoder = patched_jsonable_encoder

from ekm.core.mesh import EKM
from ekm.core.agent import EKMAgent
from ekm.storage.sql import SQLStorage
from ekm.providers.gemini import GeminiProvider
from ekm.core.models import Base, Workspace, GKU, AKU, AKURelationship, gku_aku_association, ChatSession, ChatMessage, Persona, ReflectiveConsciousness, Setting, Task, TaskStatus
from ekm.core.consolidation import SleepConsolidator
from ekm.utils.document_loader import DocumentLoader

app = FastAPI(title="EKM Sidecar API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DB_URL = os.environ.get("EKM_DB_URL", "sqlite:///ekm.db")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize EKM Components
engine = create_engine(DB_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_ekm_instance(workspace_id: Optional[str] = None):
    db_session = SessionLocal()
    storage = SQLStorage(db=db_session)
    api_key = GEMINI_API_KEY
    provider = GeminiProvider(api_key=api_key)

    # Default config
    config = {
        "EKM_SEMANTIC_THRESHOLD": 0.70,
        "VECTOR_DIMENSION": 3072,
        "EKM_MIN_CHUNK_SIZE": 512,
        "EKM_MAX_CHUNK_SIZE": 2048,
        "RL_LEARNING_RATE": 0.1
    }

    # Override with database settings if workspace exists
    if workspace_id:
        try:
            ws_uuid = uuid.UUID(workspace_id) if isinstance(workspace_id, str) else workspace_id
            settings = db_session.query(Setting).filter(Setting.workspace_id == ws_uuid).all()
            for s in settings:
                config[s.key] = s.value
        except Exception as e:
            logger.warning(f"Error loading settings for workspace {workspace_id}: {e}")

    return EKM(storage=storage, llm=provider, embeddings=provider, config=config), db_session

class SessionCreate(BaseModel):
    workspace_id: str
    name: Optional[str] = "New Chat"

class ChatRequest(BaseModel):
    workspace_id: str
    message: str
    session_id: Optional[str] = None
    include_chain_of_thoughts: Optional[bool] = False
    use_agentic_system: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    mode_used: str
    workspace_id: str
    metadata: Optional[dict] = None
    chain_of_thoughts: Optional[str] = None

class WorkspaceInfo(BaseModel):
    id: str
    name: str

class PersonaUpdate(BaseModel):
    name: str
    personality: Optional[str] = None
    voice_style: Optional[str] = None

class PersonaInfo(BaseModel):
    name: str
    personality: Optional[str]
    voice_style: Optional[str]

class SettingInfo(BaseModel):
    key: str
    value: Any

class SettingsResponse(BaseModel):
    settings: Dict[str, Any]

class DeepResearchRequest(BaseModel):
    workspace_id: str
    query: str
    max_iterations: Optional[int] = 3

class TaskCreateRequest(BaseModel):
    workspace_id: str
    name: str
    description: str
    task_type: str  # 'deep_research', 'training', etc.

class TaskResponse(BaseModel):
    id: str
    name: str
    description: str
    status: str
    progress: float
    result: Optional[str] = None
    error: Optional[str] = None

@app.get("/workspaces", response_model=List[WorkspaceInfo])
async def list_workspaces():
    _, db = get_ekm_instance()
    try:
        workspaces = db.query(Workspace).all()
        return [{"id": str(w.id), "name": w.name} for w in workspaces]
    finally:
        db.close()

@app.post("/workspaces", response_model=WorkspaceInfo)
async def create_workspace(name: str):
    _, db = get_ekm_instance()
    try:
        new_id = uuid.uuid4()
        workspace = Workspace(id=new_id, name=name, user_id="gui_user")
        db.add(workspace)
        db.commit()
        return {"id": str(new_id), "name": name}
    finally:
        db.close()

@app.get("/workspaces/{workspace_id}/sessions")
async def get_sessions(workspace_id: str):
    _, db = get_ekm_instance(workspace_id)
    try:
        ws_uuid = uuid.UUID(workspace_id)
        sessions = db.query(ChatSession).filter(ChatSession.workspace_id == ws_uuid).order_by(ChatSession.created_at.desc()).all()
        return [{"id": str(s.id), "name": s.name, "created_at": s.created_at} for s in sessions]
    finally:
        db.close()

@app.post("/sessions")
async def create_session(request: SessionCreate):
    _, db = get_ekm_instance(request.workspace_id)
    try:
        ws_uuid = uuid.UUID(request.workspace_id)
        new_id = uuid.uuid4()
        session = ChatSession(id=new_id, workspace_id=ws_uuid, name=request.name)
        db.add(session)
        db.commit()
        return {"id": str(new_id), "name": request.name}
    finally:
        db.close()

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    _, db = get_ekm_instance()
    try:
        sid = uuid.UUID(session_id)
        session = db.query(ChatSession).filter(ChatSession.id == sid).first()
        if session:
            db.delete(session)
            db.commit()
            return {"status": "deleted"}
        raise HTTPException(status_code=404, detail="Session not found")
    finally:
        db.close()

@app.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    _, db = get_ekm_instance()
    try:
        sid = uuid.UUID(session_id)
        messages = db.query(ChatMessage).filter(ChatMessage.session_id == sid).order_by(ChatMessage.created_at.asc()).all()
        return [
            {
                "role": m.role,
                "content": m.content,
                "mode_used": m.mode_used,
                "metadata": m.msg_metadata,
                "created_at": m.created_at
            } for m in messages
        ]
    finally:
        db.close()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    ekm, db = get_ekm_instance(request.workspace_id)
    try:
        session_id = request.session_id
        session = None
        if session_id:
            sid = uuid.UUID(session_id)
            session = db.query(ChatSession).filter(ChatSession.id == sid).first()

        # If no session provided or not found, create one (fallback)
        if not session:
            ws_uuid = uuid.UUID(request.workspace_id)
            new_id = uuid.uuid4()
            session = ChatSession(id=new_id, workspace_id=ws_uuid, name=request.message[:30] + "...")
            db.add(session)
            db.flush()
            session_id = str(new_id)

        # Initialize Agent with focus buffer from session
        agent = EKMAgent(ekm, request.workspace_id)
        if session.focus_buffer_state:
            # Actually FocusBuffer is Pydantic, so we can use model_validate or just update
            from ekm.core.state import FocusBuffer
            try:
                # Create a new FocusBuffer instance from the stored state
                agent.focus_buffer = FocusBuffer.model_validate({"items": session.focus_buffer_state})
            except Exception as e:
                logger.warning(f"Failed to parse focus buffer state: {e}")
                # Fallback: try to update items individually
                try:
                    for k, v in session.focus_buffer_state.items():
                        if isinstance(v, dict):
                            # Assuming the values are of the same type as the default
                            if agent.focus_buffer.items:
                                sample_item = next(iter(agent.focus_buffer.items.values()))
                                item_type = type(sample_item)
                                agent.focus_buffer.items[k] = item_type(**v)
                            else:
                                # If no items exist, we can't determine the type, so store as-is
                                agent.focus_buffer.items[k] = v
                        else:
                            agent.focus_buffer.items[k] = v
                except Exception as fallback_error:
                    logger.error(f"Focus buffer fallback also failed: {fallback_error}")

        # Fetch recent history for context
        history_msgs = db.query(ChatMessage).filter(ChatMessage.session_id == session.id).order_by(ChatMessage.created_at.desc()).limit(10).all()
        agent.history = [{"role": m.role, "content": m.content} for m in reversed(history_msgs)]

        # Load persona and reflection if available
        persona_data = db.query(Persona).filter(Persona.workspace_id == uuid.UUID(request.workspace_id)).first()
        consciousness_data = db.query(ReflectiveConsciousness).filter(ReflectiveConsciousness.workspace_id == uuid.UUID(request.workspace_id)).order_by(ReflectiveConsciousness.created_at.desc()).first()

        # Pass to agent (we'll need to update EKMAgent.chat to accept these or set them)
        agent.persona = {
            "name": persona_data.name if persona_data else "EKM Assistant",
            "personality": persona_data.personality if persona_data else "helpful, curious",
            "voice_style": persona_data.voice_style if persona_data else "professional"
        }
        agent.current_consciousness = consciousness_data

        result = await agent.chat(
            request.message, 
            include_chain_of_thoughts=request.include_chain_of_thoughts,
            use_agentic_system=request.use_agentic_system
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Save messages to history
        user_msg = ChatMessage(session_id=session.id, role='user', content=request.message)
        asst_msg = ChatMessage(
            session_id=session.id,
            role='assistant',
            content=result["response"],
            mode_used=result["mode_used"],
            msg_metadata=result.get("metadata") or {}
        )
        db.add(user_msg)
        db.add(asst_msg)

        # Update session name if it was "New Chat"
        if session.name == "New Chat":
            session.name = request.message[:40]

        # Persist FocusBuffer state
        session.focus_buffer_state = {k: v.model_dump() for k, v in agent.focus_buffer.items.items()}

        db.commit()

        return {
            "response": result["response"],
            "mode_used": result["mode_used"],
            "workspace_id": request.workspace_id,
            "metadata": result.get("metadata"),
            "chain_of_thoughts": result.get("chain_of_thoughts")
        }
    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/workspaces/{workspace_id}/persona", response_model=PersonaInfo)
async def get_persona(workspace_id: str):
    _, db = get_ekm_instance(workspace_id)
    try:
        ws_uuid = uuid.UUID(workspace_id)
        persona = db.query(Persona).filter(Persona.workspace_id == ws_uuid).first()
        if not persona:
            # Return a default persona
            return {"name": "EKM Assistant", "personality": "helpful, curious", "voice_style": "professional"}
        return {
            "name": persona.name,
            "personality": persona.personality,
            "voice_style": persona.voice_style
        }
    finally:
        db.close()

@app.put("/workspaces/{workspace_id}/persona", response_model=PersonaInfo)
async def update_persona(workspace_id: str, request: PersonaUpdate):
    _, db = get_ekm_instance(workspace_id)
    try:
        ws_uuid = uuid.UUID(workspace_id)
        persona = db.query(Persona).filter(Persona.workspace_id == ws_uuid).first()
        if not persona:
            persona = Persona(workspace_id=ws_uuid, name=request.name)
            db.add(persona)

        persona.name = request.name
        persona.personality = request.personality
        persona.voice_style = request.voice_style
        db.commit()
        return {
            "name": persona.name,
            "personality": persona.personality,
            "voice_style": persona.voice_style
        }
    finally:
        db.close()

@app.get("/workspaces/{workspace_id}/consciousness")
async def get_consciousness(workspace_id: str):
    _, db = get_ekm_instance(workspace_id)
    try:
        ws_uuid = uuid.UUID(workspace_id)
        consciousness = db.query(ReflectiveConsciousness).filter(ReflectiveConsciousness.workspace_id == ws_uuid).order_by(ReflectiveConsciousness.created_at.desc()).first()
        if not consciousness:
            return {"mood": "Stable", "thought_summary": "I am ready to help."}
        return {
            "mood": consciousness.mood or "Stable",
            "thought_summary": consciousness.thought_summary or "I am processing knowledge.",
            "focus_topics": consciousness.focus_topics or [],
            "created_at": consciousness.created_at
        }
    finally:
        db.close()

@app.get("/workspaces/{workspace_id}/settings")
async def get_settings(workspace_id: str):
    ekm, db = get_ekm_instance(workspace_id)
    try:
        # get_ekm_instance already merges DB settings into the default config
        return {"settings": ekm.config}
    finally:
        db.close()

@app.put("/workspaces/{workspace_id}/settings")
async def update_settings(workspace_id: str, settings: Dict[str, Any]):
    _, db = get_ekm_instance(workspace_id)
    try:
        ws_uuid = uuid.UUID(workspace_id)
        for key, value in settings.items():
            setting = db.query(Setting).filter(Setting.workspace_id == ws_uuid, Setting.key == key).first()
            if not setting:
                setting = Setting(workspace_id=ws_uuid, key=key, value=value)
                db.add(setting)
            else:
                setting.value = value
        db.commit()
        return {"status": "updated"}
    finally:
        db.close()

@app.post("/train/{workspace_id}")
async def train(workspace_id: str, background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    ekm, db = get_ekm_instance(workspace_id)
    try:
        loader = DocumentLoader(llm_provider=ekm.llm)
        contents_to_train = []

        for file in files:
            file_content = await file.read()
            try:
                text_content = await loader.load_bytes(file_content, file.filename)
                contents_to_train.append(text_content)
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                continue

        async def run_training():
            for content in contents_to_train:
                await ekm.train(workspace_id, content)

        background_tasks.add_task(run_training)
        return {"status": "Processing", "file_count": len(contents_to_train)}
    finally:
        db.close()

@app.post("/sleep/{workspace_id}")
async def sleep_cycle(workspace_id: str):
    ekm, db = get_ekm_instance(workspace_id)
    try:
        ws_uuid = uuid.UUID(workspace_id)
        api_key = GEMINI_API_KEY
        provider = GeminiProvider(api_key=api_key)
        consolidator = SleepConsolidator(ekm.storage, provider, provider)
        consolidator_results = await consolidator.run_consolidation(workspace_id)

        # --- SELF-REFLECTION ADDITION ---
        # Fetch some recent AKUs for context
        akus = db.query(AKU).filter(AKU.workspace_id == ws_uuid).order_by(AKU.created_at.desc()).limit(20).all()
        recent_context = "\n".join([a.content for a in akus])

        agent = EKMAgent(ekm, workspace_id)
        # Load persona
        persona_data = db.query(Persona).filter(Persona.workspace_id == ws_uuid).first()
        if persona_data:
            agent.persona = {
                "name": persona_data.name,
                "personality": persona_data.personality,
                "voice_style": persona_data.voice_style
            }

        reflection = await agent.reflect(recent_context)

        # Save consciousness state
        consciousness = ReflectiveConsciousness(
            workspace_id=ws_uuid,
            mood=reflection.get("mood", "Stable"),
            thought_summary=reflection.get("thought_summary", "I am processing knowledge."),
            focus_topics=reflection.get("focus_topics", [])
        )
        db.add(consciousness)
        db.commit()

        return {
            "status": "Complete",
            "consolidator_results": consolidator_results,
            "reflection": reflection
        }
    finally:
        db.close()


# ────────────────────────────────────────────────────────────
#  Deep Research & Task Management Endpoints
# ────────────────────────────────────────────────────────────

@app.post("/deep_research/{workspace_id}")
async def deep_research(request: DeepResearchRequest):
    """Initiate a deep research task that generates a LaTeX PDF based on the query."""
    ekm, db = get_ekm_instance(request.workspace_id)
    try:
        agent = EKMAgent(ekm, request.workspace_id)
        
        # Load persona
        ws_uuid = uuid.UUID(request.workspace_id)
        persona_data = db.query(Persona).filter(Persona.workspace_id == ws_uuid).first()
        if persona_data:
            agent.persona = {
                "name": persona_data.name,
                "personality": persona_data.personality,
                "voice_style": persona_data.voice_style
            }

        # Perform deep research
        result = await agent.generate_deep_research_pdf(request.query, max_iterations=request.max_iterations)
        
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/tasks")
async def create_task(request: TaskCreateRequest):
    """Create a new task in the task manager."""
    ekm, db = get_ekm_instance(request.workspace_id)
    try:
        agent = EKMAgent(ekm, request.workspace_id)
        
        # Create task in the task manager
        task_id = agent.task_manager.create_task(
            name=request.name,
            description=request.description,
            task_metadata={"task_type": request.task_type, "workspace_id": request.workspace_id}
        )
        
        # For certain task types, we can start them immediately
        if request.task_type == "deep_research":
            # In a real implementation, we would queue this task
            # For now, we'll just return the task info
            pass
        
        task = agent.task_manager.get_task(task_id)
        
        return TaskResponse(
            id=task.id,
            name=task.name,
            description=task.description,
            status=task.status.value,
            progress=task.progress,
            result=task.result,
            error=task.error
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get the status of a specific task."""
    # For now, we'll return a mock response
    # In a real implementation, tasks would be stored in the database
    # and we would retrieve the task by ID
    return {
        "id": task_id,
        "status": "completed",  # Mock status
        "progress": 1.0,       # Mock progress
        "result": "Task completed successfully"
    }


@app.get("/tasks/workspace/{workspace_id}")
async def get_workspace_tasks(workspace_id: str):
    """Get all tasks for a specific workspace."""
    _, db = get_ekm_instance(workspace_id)
    try:
        ws_uuid = uuid.UUID(workspace_id)
        tasks = db.query(Task).filter(Task.workspace_id == ws_uuid).order_by(Task.created_at.desc()).all()
        
        return [{
            "id": str(task.id),
            "name": task.name,
            "description": task.description,
            "status": task.status,
            "progress": task.progress,
            "result": task.result,
            "error": task.error,
            "created_at": task.created_at,
            "updated_at": task.updated_at
        } for task in tasks]
    finally:
        db.close()


@app.post("/tasks/deep_research")
async def create_deep_research_task(request: TaskCreateRequest):
    """Create a deep research task."""
    ekm, db = get_ekm_instance(request.workspace_id)
    try:
        # Create the task record in the database
        task_db = Task(
            workspace_id=uuid.UUID(request.workspace_id),
            name=request.name,
            description=request.description,
            status=TaskStatus.PENDING.value,
            progress=0.0,
            task_metadata={"task_type": request.task_type}
        )
        db.add(task_db)
        db.commit()
        db.refresh(task_db)
        
        # In a real implementation, we would queue this task for background processing
        # For now, we'll return the task info
        return TaskResponse(
            id=str(task_db.id),
            name=task_db.name,
            description=task_db.description,
            status=task_db.status,
            progress=task_db.progress,
            result=task_db.result,
            error=task_db.error
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# ────────────────────────────────────────────────────────────
#  Graph Visualization Endpoint
# ────────────────────────────────────────────────────────────

def _deserialize_embedding(emb):
    """Safely deserialize an embedding to a list of floats."""
    if emb is None:
        return None
    if isinstance(emb, str):
        try:
            return json.loads(emb)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(emb, (bytes, bytearray)):
        try:
            return json.loads(emb.decode("utf-8"))
        except:
            return None
    if isinstance(emb, (list, tuple)):
        return list(emb)
    if hasattr(emb, "tolist"):
        return emb.tolist()
    return None


def _project_to_3d(embeddings: List[List[float]], scale: float = 8.0) -> List[List[float]]:
    """Project high-dimensional embeddings to 3D using PCA. Falls back to random."""
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [[0.0, 0.0, 0.0]]

    mat = np.array(embeddings, dtype=np.float32)
    std = mat.std(axis=0)
    valid = std > 1e-8
    if valid.sum() < 3:
        rng = np.random.default_rng(42)
        return (rng.standard_normal((n, 3)) * scale).tolist()

    mat = mat[:, valid]
    mat = mat - mat.mean(axis=0)

    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(3, mat.shape[1]), random_state=42)
        projected = pca.fit_transform(mat)
    except ImportError:
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        projected = U[:, :3] * S[:3]

    if projected.shape[1] < 3:
        pad = np.zeros((n, 3 - projected.shape[1]))
        projected = np.hstack([projected, pad])

    max_abs = np.abs(projected).max()
    if max_abs > 0:
        projected = projected / max_abs * scale

    return projected.tolist()


@app.get("/graph/{workspace_id}")
async def get_graph(workspace_id: str):
    """Return the full knowledge graph for 3D visualization."""
    _, db = get_ekm_instance(workspace_id)
    try:
        ws_uuid = uuid.UUID(workspace_id)

        # --- Fetch AKUs ---
        akus = db.query(AKU).filter(AKU.workspace_id == ws_uuid).all()
        aku_embeddings = []
        aku_ids = []
        aku_has_embedding = {}
        for a in akus:
            emb = _deserialize_embedding(a.embedding)
            if emb and len(emb) > 0:
                aku_embeddings.append(emb)
                aku_has_embedding[str(a.id)] = len(aku_embeddings) - 1
            aku_ids.append(str(a.id))

        # --- Fetch GKUs ---
        gkus = db.query(GKU).filter(GKU.workspace_id == ws_uuid).all()
        gku_embeddings = []
        gku_has_embedding = {}
        for g in gkus:
            emb = _deserialize_embedding(g.centroid_embedding)
            if emb and len(emb) > 0:
                gku_embeddings.append(emb)
                gku_has_embedding[str(g.id)] = len(gku_embeddings) - 1

        # --- Project to 3D ---
        all_embeddings = aku_embeddings + gku_embeddings
        positions_3d = _project_to_3d(all_embeddings)

        rng = np.random.default_rng(7)

        nodes = []
        for a in akus:
            aid = str(a.id)
            if aid in aku_has_embedding:
                idx = aku_has_embedding[aid]
                pos = positions_3d[idx]
            else:
                pos = (rng.standard_normal(3) * 6).tolist()
            nodes.append({
                "id": aid,
                "type": "aku",
                "label": (a.content or "")[:80],
                "content": a.content or "",
                "x": round(pos[0], 3),
                "y": round(pos[1], 3),
                "z": round(pos[2], 3),
                "archived": bool(a.is_archived),
            })

        gku_offset = len(aku_embeddings)
        for g in gkus:
            gid = str(g.id)
            if gid in gku_has_embedding:
                idx = gku_offset + gku_has_embedding[gid]
                pos = positions_3d[idx]
            else:
                pos = (rng.standard_normal(3) * 4).tolist()
            nodes.append({
                "id": gid,
                "type": "gku",
                "label": g.name or "Unnamed Concept",
                "content": g.name or "",
                "x": round(pos[0], 3),
                "y": round(pos[1], 3),
                "z": round(pos[2], 3),
                "pattern": g.pattern_signature,
                "member_count": len(g.member_akus) if hasattr(g, 'member_akus') else 0,
            })

        # --- Fetch edges (AKU relationships) ---
        rels = db.query(AKURelationship).filter(AKURelationship.workspace_id == ws_uuid).all()
        edges = []
        for r in rels:
            edge_type = "semantic"
            weight = r.semantic_similarity or 0.0
            if (r.causal_weight or 0) > weight:
                edge_type = "causal"
                weight = r.causal_weight
            if (r.temporal_proximity or 0) > weight:
                edge_type = "temporal"
                weight = r.temporal_proximity
            edges.append({
                "source": str(r.source_aku_id),
                "target": str(r.target_aku_id),
                "weight": round(weight, 3),
                "type": edge_type,
                "semantic": round(r.semantic_similarity or 0, 3),
                "causal": round(r.causal_weight or 0, 3),
                "temporal": round(r.temporal_proximity or 0, 3),
            })

        # --- GKU → AKU membership edges ---
        assoc_rows = db.query(gku_aku_association).all()
        for row in assoc_rows:
            edges.append({
                "source": str(row.gku_id),
                "target": str(row.aku_id),
                "weight": 1.0,
                "type": "membership",
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "aku_count": len(akus),
                "gku_count": len(gkus),
                "edge_count": len(edges),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)