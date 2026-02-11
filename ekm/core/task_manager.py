"""
Task Manager System for EKM - Manages research tasks and workflows efficiently with persistence.
"""
import asyncio
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field

from sqlalchemy.orm import Session
from .models import Task as DBTask, TaskStatus

logger = logging.getLogger(__name__)

# Re-exporting Task dataclass for compatibility if needed, 
# but we should primarily use DBTask or a Pydantic model in the app layer.
@dataclass
class Task:
    """Represents a single task, mirrored from DB for in-memory use if needed."""
    id: str
    name: str
    description: str
    status: Union[TaskStatus, str]
    created_at: datetime
    updated_at: datetime
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: float = 0.0
    task_metadata: Dict[str, Any] = field(default_factory=dict)
    workspace_id: Optional[str] = None

    @classmethod
    def from_db(cls, db_task: DBTask):
        return cls(
            id=str(db_task.id),
            name=db_task.name,
            description=db_task.description,
            status=db_task.status,
            created_at=db_task.created_at,
            updated_at=db_task.updated_at,
            result=db_task.result,
            error=db_task.error,
            progress=db_task.progress,
            task_metadata=db_task.task_metadata or {},
            workspace_id=str(db_task.workspace_id)
        )

class TaskManager:
    """Manages asynchronous tasks for the EKM system with database persistence."""
    
    def __init__(self, db_session: Optional[Session] = None, workspace_id: Optional[str] = None):
        self.db = db_session
        self.workspace_id = workspace_id
        # Callbacks remain in-memory as they are runtime specific
        self._callbacks: Dict[str, List[Callable]] = {}
    
    def _get_db_task(self, task_id: str) -> Optional[DBTask]:
        if not self.db: return None
        try:
            return self.db.query(DBTask).filter(DBTask.id == uuid.UUID(task_id)).first()
        except ValueError:
            return None

    def create_task(self, name: str, description: str, task_metadata: Dict[str, Any] = None) -> str:
        """Create a new task in the database and return its ID."""
        task_id = str(uuid.uuid4())
        metadata = task_metadata or {}
        
        # Ensure workspace_id is present
        ws_id = metadata.get('workspace_id') or self.workspace_id
        if not ws_id:
            raise ValueError("Workspace ID required to create a task")

        if self.db:
            db_task = DBTask(
                id=uuid.UUID(task_id),
                workspace_id=uuid.UUID(str(ws_id)),
                name=name,
                description=description,
                status=TaskStatus.PENDING.value,
                progress=0.0,
                task_metadata=metadata,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            self.db.add(db_task)
            self.db.commit()
            self.db.refresh(db_task)
        else:
            logger.warning("TaskManager initialized without DB session. Task will not be persisted.")
            
        self._callbacks[task_id] = []
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by its ID."""
        if self.db:
            db_task = self._get_db_task(task_id)
            if db_task:
                return Task.from_db(db_task)
        return None
    
    def get_all_tasks(self, limit: int = 50) -> List[Task]:
        """Get all tasks for the current workspace."""
        if self.db and self.workspace_id:
            db_tasks = self.db.query(DBTask).filter(
                DBTask.workspace_id == uuid.UUID(self.workspace_id)
            ).order_by(DBTask.updated_at.desc()).limit(limit).all()
            return [Task.from_db(t) for t in db_tasks]
        return []
    
    def update_task_status(self, task_id: str, status: Union[TaskStatus, str]):
        """Update the status of a task."""
        if self.db:
            db_task = self._get_db_task(task_id)
            if db_task:
                # Handle enum or string
                status_val = status.value if isinstance(status, TaskStatus) else status
                db_task.status = status_val
                db_task.updated_at = datetime.utcnow()
                self.db.commit()
    
    def update_task_progress(self, task_id: str, progress: float):
        """Update the progress of a task (0.0 to 1.0)."""
        if self.db:
            db_task = self._get_db_task(task_id)
            if db_task:
                db_task.progress = max(0.0, min(1.0, progress))
                db_task.updated_at = datetime.utcnow()
                self.db.commit()
    
    def set_task_result(self, task_id: str, result: Any):
        """Set the result of a task."""
        if self.db:
            db_task = self._get_db_task(task_id)
            if db_task:
                # Serialize result if needed, though DB column is Text
                # Ideally DB column should be JSON or we stringify here
                result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                db_task.result = result_str
                db_task.updated_at = datetime.utcnow()
                self.db.commit()
    
    def set_task_error(self, task_id: str, error: str):
        """Set an error for a task."""
        if self.db:
            db_task = self._get_db_task(task_id)
            if db_task:
                db_task.error = error
                db_task.status = TaskStatus.FAILED.value
                db_task.updated_at = datetime.utcnow()
                self.db.commit()
    
    def add_callback(self, task_id: str, callback: Callable):
        """Add a callback to be called when the task completes."""
        if task_id not in self._callbacks:
            self._callbacks[task_id] = []
        self._callbacks[task_id].append(callback)
    
    async def run_task_async(self, task_id: str, coro_func: Callable, *args, **kwargs) -> Optional[Task]:
        """Run an async function as a task."""
        # Note: This runs in the mapped asyncio loop, but updates DB synchronously via SQLAlchemy?
        # Ideally we'd use async session, but for now we rely on the sync session passed in.
        
        try:
            self.update_task_status(task_id, TaskStatus.RUNNING)
            
            # Run the coroutine
            result = await coro_func(*args, **kwargs)
            
            self.set_task_result(task_id, result)
            self.update_task_status(task_id, TaskStatus.COMPLETED)
            self.update_task_progress(task_id, 1.0)
            
            task = self.get_task(task_id)

            # Call callbacks
            if task_id in self._callbacks:
                for callback in self._callbacks[task_id]:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(task)
                    else:
                        callback(task)
                    
            return task
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            self.set_task_error(task_id, str(e))
            return self.get_task(task_id)
    
    def cancel_task(self, task_id: str):
        """Cancel a task."""
        self.update_task_status(task_id, TaskStatus.CANCELLED)


# Example usage:
# async def example_usage():
#     tm = TaskManager()
#     
#     # Create a task
#     task_id = tm.create_task("Research Task", "Perform deep research on a topic")
#     
#     # Define an async function to run as a task
#     async def research_function(query: str):
#         # Simulate some async work
#         await asyncio.sleep(2)
#         return f"Research completed for: {query}"
#     
#     # Run the task
#     await tm.run_task_async(task_id, research_function, "Artificial Intelligence")
#     
#     # Check the result
#     task = tm.get_task(task_id)
#     print(f"Task result: {task.result}")