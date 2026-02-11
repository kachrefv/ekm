import uuid
import logging
import json
import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ekm.core.models import Task, TaskStatus, Persona, ReflectiveConsciousness, AKU
from ekm.core.agent import EKMAgent
from ekm.core.task_manager import TaskManager
from ekm.utils.document_loader import DocumentLoader
from ekm.core.consolidation import SleepConsolidator
from ekm.providers.gemini import GeminiProvider

from ..models import TaskCreateRequest, TaskResponse, DeepResearchRequest
from ..core.dependencies import get_ekm_instance, get_db, SessionLocal
from ..core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Tasks"])

# --- Background Task Helper ---
async def run_training(workspace_id: str, contents: List[str]):
    ekm = get_ekm_instance(workspace_id)
    for content in contents:
        await ekm.train(workspace_id, content)

@router.post("/train/{workspace_id}")
async def train(workspace_id: str, background_tasks: BackgroundTasks, files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    ekm = get_ekm_instance(workspace_id)
    # Validate workspace existence indirectly or explicitly
    
    loader = DocumentLoader(llm_provider=ekm.llm)
    contents_to_train = []

    for file in files:
        file_content = await file.read()
        try:
            # DocumentLoader might expect file path or bytes, here we use bytes if supported or save tmp
            # original server.py used load_bytes
            text_content = await loader.load_bytes(file_content, file.filename)
            contents_to_train.append(text_content)
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            continue

    background_tasks.add_task(run_training, workspace_id, contents_to_train)
    return {"status": "Processing", "file_count": len(contents_to_train)}


@router.post("/sleep/{workspace_id}")
async def sleep_cycle(workspace_id: str, db: Session = Depends(get_db)):
    ekm = get_ekm_instance(workspace_id)
    ws_uuid = uuid.UUID(workspace_id)
    
    # We might need a separate provider instance if we want specific config, 
    # but EKM instance already has one.
    # Original server.py created a NEW GeminiProvider here using the env key.
    # We should reuse the EKM's provider if possible, or create new one if needed for separation.
    # reusing ekm.llm/ekm.embeddings seems safer for resource usage.
    
    # For consolidation, we need a provider.
    consolidator = SleepConsolidator(ekm.storage, ekm.llm, ekm.embeddings)
    consolidator_results = await consolidator.run_consolidation(workspace_id)

    # --- SELF-REFLECTION ADDITION ---
    # Fetch some recent AKUs for context
    akus = db.query(AKU).filter(AKU.workspace_id == ws_uuid).order_by(AKU.created_at.desc()).limit(20).all()
    recent_context = "\n".join([a.content for a in akus]) if akus else "No recent memories."

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


@router.post("/deep_research/{workspace_id}")
async def deep_research(request: DeepResearchRequest, db: Session = Depends(get_db)):
    """Initiate a deep research task that generates a LaTeX PDF based on the query."""
    ekm = get_ekm_instance(request.workspace_id)
    try:
        agent = EKMAgent(ekm, request.workspace_id)
        
        # Load persona
        ws_uuid = uuid.UUID(request.workspace_id)
        persona_data = db.query(Persona).filter(Persona.workspace_id == ws_uuid).first()
        if persona_data:
            agent.persona = {
                "name": persona_data.name,
                "personality": persona.personality,
                "voice_style": persona.voice_style
            }

        result = await agent.generate_deep_research_pdf(request.query, max_iterations=request.max_iterations)
        return result
    except Exception as e:
        import traceback
# Helper function to format TaskResponse
def get_task_response(tm: TaskManager, task_id: str) -> TaskResponse:
    t = tm.get_task(task_id)
    if not t:
        raise HTTPException(status_code=500, detail="Task creation failed or task not found")
    return TaskResponse(
        id=str(t.id),
        name=t.name,
        description=t.description,
        status=t.status if isinstance(t.status, str) else t.status.value,
        progress=t.progress,
        result=str(t.result) if t.result else None,
        error=t.error
    )

# --- Task Management Endpoints ---

@router.post("/tasks", response_model=TaskResponse)
async def create_task(request: TaskCreateRequest, db: Session = Depends(get_db)):
    """Create a new generic task."""
    tm = TaskManager(db_session=db, workspace_id=request.workspace_id)
    task_id = tm.create_task(
        name=request.name,
        description=request.description,
        task_metadata={"type": request.task_type}
    )
    
    # Check if we need to start a known background job
    # For now, just generic task creation. Logic for running it specific to type
    # would go here or be triggered separately.
    
    return get_task_response(tm, str(task_id))

@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, db: Session = Depends(get_db)):
    """Get the status of a specific task."""
    # We don't know the workspace_id here easily without querying, 
    # but TaskManager.get_task only needs DB.
    tm = TaskManager(db_session=db)
    task = tm.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskResponse(
        id=str(task.id),
        name=task.name,
        description=task.description,
        status=task.status if isinstance(task.status, str) else task.status.value,
        progress=task.progress,
        result=str(task.result) if task.result else None,
        error=task.error
    )

@router.get("/workspaces/{workspace_id}/tasks", response_model=List[TaskResponse])
async def list_tasks(workspace_id: str, db: Session = Depends(get_db)):
    """List all tasks for a workspace."""
    tm = TaskManager(db_session=db, workspace_id=workspace_id)
    tasks = tm.get_all_tasks(limit=20)
    
    response = []
    for task in tasks:
        response.append(TaskResponse(
            id=str(task.id),
            name=task.name,
            description=task.description,
            status=task.status if isinstance(task.status, str) else task.status.value,
            progress=task.progress,
            result=str(task.result) if task.result else None,
            error=task.error
        ))
    return response

@router.get("/tasks/events")
async def tasks_events(workspace_id: str):
    """Server-Sent Events for real-time task updates."""
    
    async def event_generator():
        last_states = {}
        
        try:
            while True:
                db = SessionLocal()
                try:
                    tm = TaskManager(db_session=db, workspace_id=workspace_id)
                    tasks = tm.get_all_tasks(limit=10)
                    
                    # Serialize all fetched tasks (recent 10)
                    active_tasks = []
                    for task in tasks:
                        t_data = {
                            "id": str(task.id),
                            "name": task.name,
                            "status": task.status if isinstance(task.status, str) else task.status.value,
                            "progress": task.progress,
                            "updated_at": str(task.updated_at)
                        }
                        active_tasks.append(t_data)
                    
                    # Create a signature of the current state to detect changes
                    current_state_sig = json.dumps(active_tasks, sort_keys=True)
                    
                    # If state changed from last yield, send it
                    if current_state_sig != last_states.get('global_sig'):
                        yield f"data: {json.dumps(active_tasks)}\n\n"
                        last_states['global_sig'] = current_state_sig
                    else:
                        yield ": keep-alive\n\n"

                finally:
                    db.close()
                    
                await asyncio.sleep(2) # Poll every 2 seconds
        except asyncio.CancelledError:
            pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/deep_research/{workspace_id}")
async def deep_research(request: DeepResearchRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Start a deep research task."""
    logger.info(f"Starting deep research for: {request.query}")
    
    ekm = get_ekm_instance(request.workspace_id)
    agent = EKMAgent(ekm=ekm, workspace_id=request.workspace_id)
    
    # Inject the current request session into the agent's task manager for this request scope:
    agent.task_manager.db = db 
    
    # Create the task entry
    task_id = agent.task_manager.create_task(
        name=f"Deep Research: {request.query[:30]}...",
        description=f"Deep research on '{request.query}'",
        task_metadata={"type": "deep_research", "query": request.query}
    )
    
    # Define the worker function
    async def run_research(tid: str, query: str, workspace_id: str, max_iterations: int):
        db_bg = SessionLocal()
        try:
            ekm_bg = get_ekm_instance(workspace_id)
            
            tm_bg = TaskManager(db_session=db_bg, workspace_id=workspace_id)
            
            # Update status to running
            tm_bg.update_task_status(tid, TaskStatus.RUNNING)
            tm_bg.update_task_progress(tid, 0.1)
            
            # Fix: Temporarily patch the singleton's storage session for this background execution
            if hasattr(ekm_bg.storage, 'db'):
                ekm_bg.storage.db = db_bg
                
            agent_bg = EKMAgent(ekm=ekm_bg, workspace_id=workspace_id)
            agent_bg.task_manager = tm_bg
            
            result = await agent_bg.generate_deep_research_pdf(query, max_iterations=max_iterations)
            
            if result.get("status") == "success":
                tm_bg.set_task_result(tid, result.get("latex_content", ""))
                tm_bg.update_task_status(tid, TaskStatus.COMPLETED)
                tm_bg.update_task_progress(tid, 1.0)
            else:
                tm_bg.set_task_error(tid, result.get("latex_content", "Unknown error"))
                tm_bg.update_task_status(tid, TaskStatus.FAILED)
                
        except Exception as e:
            logger.error(f"Background research failed for task {tid}: {e}", exc_info=True)
            try:
                tm_bg.set_task_error(tid, str(e))
                tm_bg.update_task_status(tid, TaskStatus.FAILED)
            except Exception as inner_e:
                logger.error(f"Failed to update task {tid} with error: {inner_e}")
        finally:
            db_bg.close()

    background_tasks.add_task(run_research, str(task_id), request.query, request.workspace_id, request.max_iterations)
    
    return get_task_response(agent.task_manager, str(task_id))
