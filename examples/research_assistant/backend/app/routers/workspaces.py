import uuid
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ekm.core.models import Workspace, Persona, ReflectiveConsciousness, Setting
from ..models import WorkspaceInfo, PersonaInfo, PersonaUpdate
from ..core.dependencies import get_ekm_instance, get_db

router = APIRouter(prefix="/workspaces", tags=["Workspaces"])

@router.get("", response_model=List[WorkspaceInfo])
async def list_workspaces(db: Session = Depends(get_db)):
    workspaces = db.query(Workspace).all()
    return [{"id": str(w.id), "name": w.name} for w in workspaces]

@router.post("", response_model=WorkspaceInfo)
async def create_workspace(name: str, db: Session = Depends(get_db)):
    new_id = uuid.uuid4()
    # Ensure EKM instance is primed (though not strictly needed for just DB insert)
    get_ekm_instance() 
    
    workspace = Workspace(id=new_id, name=name, user_id="gui_user")
    db.add(workspace)
    db.commit()
    return {"id": str(new_id), "name": name}

@router.get("/{workspace_id}/persona", response_model=PersonaInfo)
async def get_persona(workspace_id: str, db: Session = Depends(get_db)):
    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID")
        
    persona = db.query(Persona).filter(Persona.workspace_id == ws_uuid).first()
    if not persona:
        return {"name": "EKM Assistant", "personality": "helpful, curious", "voice_style": "professional"}
    return {
        "name": persona.name,
        "personality": persona.personality,
        "voice_style": persona.voice_style
    }

@router.put("/{workspace_id}/persona", response_model=PersonaInfo)
async def update_persona(workspace_id: str, request: PersonaUpdate, db: Session = Depends(get_db)):
    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID")

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

@router.get("/{workspace_id}/consciousness")
async def get_consciousness(workspace_id: str, db: Session = Depends(get_db)):
    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID")

    consciousness = db.query(ReflectiveConsciousness).filter(ReflectiveConsciousness.workspace_id == ws_uuid).order_by(ReflectiveConsciousness.created_at.desc()).first()
    if not consciousness:
        return {"mood": "Stable", "thought_summary": "I am ready to help."}
    return {
        "mood": consciousness.mood or "Stable",
        "thought_summary": consciousness.thought_summary or "I am processing knowledge.",
        "focus_topics": consciousness.focus_topics or [],
        "created_at": consciousness.created_at
    }

@router.get("/{workspace_id}/settings")
async def get_settings(workspace_id: str):
    # Retrieve EKM instance which auto-loads settings
    ekm = get_ekm_instance(workspace_id)
    return {"settings": ekm.config}

@router.put("/{workspace_id}/settings")
async def update_settings(workspace_id: str, settings: Dict[str, Any], db: Session = Depends(get_db)):
    try:
        ws_uuid = uuid.UUID(workspace_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID")

    for key, value in settings.items():
        setting = db.query(Setting).filter(Setting.workspace_id == ws_uuid, Setting.key == key).first()
        if not setting:
            setting = Setting(workspace_id=ws_uuid, key=key, value=value)
            db.add(setting)
        else:
            setting.value = value
    db.commit()
    
    # Force refresh of EKM config for this workspace next time it's requested
    # (The Singleton implementation currently reloads on every call if workspace_id is passed, 
    # so this is handled implicitly, but explicit validation/reload logic could be added to dependencies.py)
    
    return {"status": "updated"}
