import uuid
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from ekm.core.models import ChatSession, ChatMessage, Persona, ReflectiveConsciousness
from ekm.core.agent import EKMAgent
from ekm.core.state import FocusBuffer

from ..models import ChatRequest, ChatResponse, SessionCreate
from ..core.dependencies import get_ekm_instance, get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

@router.get("/workspaces/{workspace_id}/sessions")
async def get_sessions(workspace_id: str, db: Session = Depends(get_db)):
    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID")
        
    sessions = db.query(ChatSession).filter(ChatSession.workspace_id == ws_uuid).order_by(ChatSession.created_at.desc()).all()
    return [{"id": str(s.id), "name": s.name, "created_at": s.created_at} for s in sessions]

@router.post("/sessions")
async def create_session(request: SessionCreate, db: Session = Depends(get_db)):
    # Ensure EKM backend is ready 
    get_ekm_instance(request.workspace_id)
    
    try:
        ws_uuid = uuid.UUID(request.workspace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID")
        
    new_id = uuid.uuid4()
    session = ChatSession(id=new_id, workspace_id=ws_uuid, name=request.name)
    db.add(session)
    db.commit()
    return {"id": str(new_id), "name": request.name}

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID")
        
    session = db.query(ChatSession).filter(ChatSession.id == sid).first()
    if session:
        db.delete(session)
        db.commit()
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

@router.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str, db: Session = Depends(get_db)):
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID")
        
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

def _restore_agent_state(agent: EKMAgent, session: ChatSession):
    """Helper to restore FocusBuffer state safely."""
    if not session.focus_buffer_state:
        return

    try:
        # Create a new FocusBuffer instance from the stored state
        agent.focus_buffer = FocusBuffer.model_validate({"items": session.focus_buffer_state})
    except Exception as e:
        logger.warning(f"Failed to parse focus buffer state: {e}")
        # Fallback: try to update items individually
        try:
            for k, v in session.focus_buffer_state.items():
                if isinstance(v, dict) and agent.focus_buffer.items:
                    # Attempt to guess type from existing items or defaul generic dict behavior
                    # This is a bit heuristic, mirroring original logic
                    agent.focus_buffer.items[k] = v
                else:
                    agent.focus_buffer.items[k] = v
        except Exception as fallback_error:
            logger.error(f"Focus buffer fallback also failed: {fallback_error}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    ekm = get_ekm_instance(request.workspace_id)
    
    try:
        session_id = request.session_id
        session = None
        if session_id:
            try:
                sid = uuid.UUID(session_id)
                session = db.query(ChatSession).filter(ChatSession.id == sid).first()
            except ValueError:
                pass # Invalid UUID passed

        # Validating workspace_id
        try:
            ws_uuid = uuid.UUID(request.workspace_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Workspace UUID")

        # If no session provided or not found, create one (fallback)
        if not session:
            new_id = uuid.uuid4()
            session = ChatSession(id=new_id, workspace_id=ws_uuid, name=request.message[:30] + "...")
            db.add(session)
            db.flush() # Flush to get ID but don't commit yet
            session_id = str(new_id)

        # Initialize Agent
        agent = EKMAgent(ekm, request.workspace_id)
        
        # Restore state
        _restore_agent_state(agent, session)

        # Fetch recent history for context
        history_msgs = db.query(ChatMessage).filter(ChatMessage.session_id == session.id).order_by(ChatMessage.created_at.desc()).limit(10).all()
        agent.history = [{"role": m.role, "content": m.content} for m in reversed(history_msgs)]

        # Load persona and reflection
        persona_data = db.query(Persona).filter(Persona.workspace_id == ws_uuid).first()
        consciousness_data = db.query(ReflectiveConsciousness).filter(ReflectiveConsciousness.workspace_id == ws_uuid).order_by(ReflectiveConsciousness.created_at.desc()).first()

        agent.persona = {
            "name": persona_data.name if persona_data else "EKM Assistant",
            "personality": persona_data.personality if persona_data else "helpful, curious",
            "voice_style": persona_data.voice_style if persona_data else "professional"
        }
        agent.current_consciousness = consciousness_data

        # Generate response
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
        # FocusBuffer items are Pydantic models, we need to dump them to dicts for JSON storage
        # The key assumption is that FocusBuffer.items is a Dict[str, Any] or Dict[str, Model]
        if hasattr(agent.focus_buffer, 'items'):
             session.focus_buffer_state = {
                 k: (v.model_dump() if hasattr(v, 'model_dump') else v) 
                 for k, v in agent.focus_buffer.items.items()
             }

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
