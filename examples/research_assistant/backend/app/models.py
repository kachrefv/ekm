from typing import Optional, Any, Dict, List
from pydantic import BaseModel

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
