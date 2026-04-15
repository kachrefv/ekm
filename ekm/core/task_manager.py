"""
Task Manager System for EKM - Manages research tasks and workflows efficiently with persistence.
"""
import asyncio
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from .models import Task as DBTask, TaskStatus, Job, JobStatus, TaskEvent

logger = logging.getLogger(__name__)


class EventSystem:
    """Event system for task and job lifecycle events."""
    
    def __init__(self, db_session: AsyncSession = None):
        self.db = db_session
        self._subscribers: Dict[str, List[Callable]] = {}
    
    async def emit(
        self,
        task_id: str,
        event_type: str,
        event_data: Optional[Dict] = None,
        job_id: Optional[str] = None
    ):
        """Emit an event for a task."""
        # Store event in database if db session available
        if self.db:
            from .models import TaskEvent
            event = TaskEvent(
                id=uuid.uuid4(),
                task_id=uuid.UUID(task_id),
                job_id=uuid.UUID(job_id) if job_id else None,
                event_type=event_type,
                event_data=event_data or {},
                created_at=datetime.utcnow()
            )
            self.db.add(event)
            await self.db.commit()
        
        # Notify subscribers
        subscribers = self._subscribers.get(event_type, [])
        for subscriber in subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(task_id, event_type, event_data, job_id)
                else:
                    subscriber(task_id, event_type, event_data, job_id)
            except Exception as e:
                logger.error(f"Event subscriber error: {e}", exc_info=True)
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to events of a specific type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from events."""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)
    
    def clear_subscribers(self, event_type: str = None):
        """Clear subscribers for an event type or all events."""
        if event_type:
            self._subscribers[event_type] = []
        else:
            self._subscribers.clear()


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
    
    def __init__(
        self,
        db_session: Optional[AsyncSession] = None,
        workspace_id: Optional[str] = None,
        event_system: Optional[EventSystem] = None
    ):
        self.db = db_session
        self.workspace_id = workspace_id
        # Callbacks remain in-memory as they are runtime specific (deprecated, use event_system)
        self._callbacks: Dict[str, List[Callable]] = {}
        self.event_system = event_system or EventSystem(db_session)
    
    async def _get_db_task(self, task_id: str) -> Optional[DBTask]:
        if not self.db:
            return None
        try:
            result = await self.db.execute(
                select(DBTask).where(DBTask.id == uuid.UUID(task_id))
            )
            return result.scalar_one_or_none()
        except ValueError:
            return None

    async def _get_db_job(self, task_id: str) -> Optional[Job]:
        if not self.db:
            return None
        try:
            result = await self.db.execute(
                select(Job).where(Job.task_id == uuid.UUID(task_id))
            )
            return result.scalar_one_or_none()
        except ValueError:
            return None

    async def create_task(self, name: str, description: str, task_metadata: Dict[str, Any] = None) -> str:
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
            
            # Create corresponding job for queue execution
            queue_name = metadata.get('queue_name', 'default')
            priority = metadata.get('priority', 0)
            max_attempts = metadata.get('max_attempts', 3)
            scheduled_at = metadata.get('scheduled_at', datetime.utcnow())
            
            job = Job(
                id=uuid.uuid4(),
                task_id=uuid.UUID(task_id),
                status=JobStatus.PENDING.value,
                queue_name=queue_name,
                priority=priority,
                max_attempts=max_attempts,
                scheduled_at=scheduled_at,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            self.db.add(job)
            
            await self.db.commit()
            await self.db.refresh(db_task)
        else:
            logger.warning("TaskManager initialized without DB session. Task will not be persisted.")

        self._callbacks[task_id] = []
        # Emit task created event
        await self.event_system.emit(task_id, 'task_created', {
            'name': name,
            'description': description,
            'metadata': metadata
        })
        return task_id
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by its ID."""
        if self.db:
            db_task = await self._get_db_task(task_id)
            if db_task:
                return Task.from_db(db_task)
        return None
    
    async def get_all_tasks(self, limit: int = 50) -> List[Task]:
        """Get all tasks for the current workspace."""
        if self.db and self.workspace_id:
            result = await self.db.execute(
                select(DBTask)
                .where(DBTask.workspace_id == uuid.UUID(self.workspace_id))
                .order_by(DBTask.updated_at.desc())
                .limit(limit)
            )
            db_tasks = result.scalars().all()
            return [Task.from_db(t) for t in db_tasks]
        return []
    
    async def update_task_status(self, task_id: str, status: Union[TaskStatus, str]):
        """Update the status of a task."""
        if self.db:
            db_task = await self._get_db_task(task_id)
            if db_task:
                # Handle enum or string
                status_val = status.value if isinstance(status, TaskStatus) else status
                db_task.status = status_val
                db_task.version += 1
                db_task.updated_at = datetime.utcnow()
                
                # Update corresponding job status if job exists
                job = await self._get_db_job(task_id)
                if job:
                    # Map TaskStatus to JobStatus (same string values for basic statuses)
                    job.status = status_val
                    job.updated_at = datetime.utcnow()
                
                await self.db.commit()
    
    async def update_task_progress(self, task_id: str, progress: float):
        """Update the progress of a task (0.0 to 1.0)."""
        if self.db:
            db_task = await self._get_db_task(task_id)
            if db_task:
                db_task.progress = max(0.0, min(1.0, progress))
                db_task.version += 1
                db_task.updated_at = datetime.utcnow()
                await self.db.commit()
    
    def _serialize_result(self, result: Any) -> Any:
        """Convert any Python object to a JSON-serializable representation."""
        if isinstance(result, (dict, list, str, int, float, bool, type(None))):
            return result
        # Try to serialize via json.dumps with default=str
        try:
            # This will convert many objects to string representation
            return json.loads(json.dumps(result, default=str))
        except Exception:
            return str(result)
    
    async def set_task_result(self, task_id: str, result: Any):
        """Set the result of a task."""
        if self.db:
            db_task = await self._get_db_task(task_id)
            if db_task:
                # Serialize result to JSON-serializable Python object
                serialized = self._serialize_result(result)
                db_task.result = serialized
                db_task.version += 1
                db_task.updated_at = datetime.utcnow()
                await self.db.commit()
    
    async def set_task_error(self, task_id: str, error: str):
        """Set an error for a task."""
        if self.db:
            db_task = await self._get_db_task(task_id)
            if db_task:
                db_task.error = error
                # Update status to FAILED and sync with job (will increment version and commit)
                await self.update_task_status(task_id, TaskStatus.FAILED)
    
    def add_callback(self, task_id: str, callback: Callable):
        """Add a callback to be called when the task completes."""
        if task_id not in self._callbacks:
            self._callbacks[task_id] = []
        self._callbacks[task_id].append(callback)
    
    async def run_task_async(self, task_id: str, coro_func: Callable, *args, **kwargs) -> Optional[Task]:
        """Run an async function as a task."""
        try:
            await self.update_task_status(task_id, TaskStatus.RUNNING)

            # Run the coroutine
            result = await coro_func(*args, **kwargs)

            await self.set_task_result(task_id, result)
            await self.update_task_status(task_id, TaskStatus.COMPLETED)
            await self.update_task_progress(task_id, 1.0)

            task = await self.get_task(task_id)

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
            await self.set_task_error(task_id, str(e))
            return await self.get_task(task_id)
    
    async def cancel_task(self, task_id: str):
        """Cancel a task."""
        await self.update_task_status(task_id, TaskStatus.CANCELLED)


class JobQueue:
    """Manages job queue operations with database persistence."""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.default_lock_timeout = 1800  # 30 minutes in seconds
    
    async def enqueue(
        self,
        task_id: str,
        queue_name: str = "default",
        priority: int = 0,
        scheduled_at: Optional[datetime] = None,
        max_attempts: int = 3
    ) -> Job:
        """Enqueue a task for execution."""
        # Check if job already exists for this task
        result = await self.db.execute(
            select(Job).where(Job.task_id == uuid.UUID(task_id))
        )
        existing_job = result.scalar_one_or_none()
        
        if existing_job:
            # Update existing job
            existing_job.queue_name = queue_name
            existing_job.priority = priority
            existing_job.scheduled_at = scheduled_at or datetime.utcnow()
            existing_job.max_attempts = max_attempts
            existing_job.status = JobStatus.PENDING.value
            existing_job.attempts = 0
            existing_job.locked_at = None
            existing_job.locked_by = None
            existing_job.next_attempt_at = None
            existing_job.updated_at = datetime.utcnow()
            job = existing_job
        else:
            # Create new job
            job = Job(
                id=uuid.uuid4(),
                task_id=uuid.UUID(task_id),
                status=JobStatus.PENDING.value,
                queue_name=queue_name,
                priority=priority,
                max_attempts=max_attempts,
                scheduled_at=scheduled_at or datetime.utcnow(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            self.db.add(job)
        
        await self.db.commit()
        await self.db.refresh(job)
        return job
    
    async def dequeue(
        self,
        queue_names: Optional[List[str]] = None,
        limit: int = 1,
        worker_id: str = None
    ) -> List[Job]:
        """Dequeue jobs for processing, locking them to prevent duplicate processing."""
        if not worker_id:
            worker_id = str(uuid.uuid4())
        
        queue_names = queue_names or ["default"]
        now = datetime.utcnow()
        
        # Find pending or retrying jobs that are not locked or have expired locks
        # and where scheduled_at <= now (and for retrying jobs, next_attempt_at <= now)
        result = await self.db.execute(
            select(Job)
            .where(
                (Job.status == JobStatus.PENDING.value) |
                (
                    (Job.status == JobStatus.RETRYING.value) &
                    (Job.next_attempt_at <= now)
                )
            )
            .where(Job.queue_name.in_(queue_names))
            .where(Job.scheduled_at <= now)
            .where(
                (Job.locked_at == None) |
                (Job.locked_at < now - timedelta(seconds=self.default_lock_timeout))
            )
            .order_by(Job.priority.asc(), Job.scheduled_at.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)  # Skip rows locked by other transactions
        )
        jobs = result.scalars().all()
        
        # Lock the jobs
        for job in jobs:
            job.status = JobStatus.RUNNING.value
            job.locked_at = now
            job.locked_by = worker_id
            job.attempts += 1
            job.last_attempt_at = now
            job.updated_at = now
        
        if jobs:
            await self.db.commit()
            for job in jobs:
                await self.db.refresh(job)
        
        return jobs
    
    async def release_lock(self, job_id: str, worker_id: str) -> bool:
        """Release lock on a job if still held by this worker."""
        result = await self.db.execute(
            select(Job).where(Job.id == uuid.UUID(job_id))
        )
        job = result.scalar_one_or_none()
        
        if not job:
            return False
        
        if job.locked_by != worker_id:
            return False
        
        # Reset lock but keep status as PENDING (or FAILED if max attempts?)
        job.locked_at = None
        job.locked_by = None
        job.updated_at = datetime.utcnow()
        
        await self.db.commit()
        return True
    
    async def complete(self, job_id: str, result: Any = None) -> bool:
        """Mark a job as completed."""
        result_obj = await self.db.execute(
            select(Job).where(Job.id == uuid.UUID(job_id))
        )
        job = result_obj.scalar_one_or_none()
        
        if not job:
            return False
        
        job.status = JobStatus.COMPLETED.value
        job.locked_at = None
        job.locked_by = None
        job.updated_at = datetime.utcnow()
        
        # Update associated task result via TaskManager
        # This will be handled separately by the worker
        
        await self.db.commit()
        return True
    
    async def fail(
        self,
        job_id: str,
        error_message: str,
        error_details: Optional[Dict] = None,
        max_attempts_exceeded: bool = False
    ) -> bool:
        """Mark a job as failed and schedule retry or move to dead letter."""
        result = await self.db.execute(
            select(Job).where(Job.id == uuid.UUID(job_id))
        )
        job = result.scalar_one_or_none()
        
        if not job:
            return False
        
        job.error_message = error_message
        job.error_details = error_details or {}
        job.updated_at = datetime.utcnow()
        
        if max_attempts_exceeded or job.attempts >= job.max_attempts:
            job.status = JobStatus.DEAD_LETTER.value
            job.locked_at = None
            job.locked_by = None
        else:
            job.status = JobStatus.RETRYING.value
            # Calculate exponential backoff: 2^attempts * base_delay (seconds)
            base_delay = 60  # 1 minute base
            delay_seconds = (2 ** (job.attempts - 1)) * base_delay
            # Add jitter ±10%
            import random
            jitter = random.uniform(0.9, 1.1)
            delay_seconds = int(delay_seconds * jitter)
            job.next_attempt_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
            job.locked_at = None
            job.locked_by = None
        
        await self.db.commit()
        return True
    
    async def retry(self, job_id: str, delay_seconds: int = 0) -> bool:
        """Schedule a job for retry."""
        result = await self.db.execute(
            select(Job).where(Job.id == uuid.UUID(job_id))
        )
        job = result.scalar_one_or_none()
        
        if not job:
            return False
        
        job.status = JobStatus.PENDING.value
        job.next_attempt_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
        job.locked_at = None
        job.locked_by = None
        job.updated_at = datetime.utcnow()
        
        await self.db.commit()
        return True


class Worker:
    """Worker that processes jobs from the queue."""
    
    def __init__(self, db_session: AsyncSession, worker_id: str = None):
        self.db = db_session
        self.worker_id = worker_id or str(uuid.uuid4())
        self.job_queue = JobQueue(db_session)
        self.task_manager = TaskManager(db_session)
        self.task_handlers: Dict[str, Callable] = {}
        self.running = False
        self.poll_interval = 5  # seconds
        self._current_tasks: Dict[str, asyncio.Task] = {}
    
    def register_handler(self, task_type: str, handler: Callable):
        """Register a handler for a specific task type."""
        self.task_handlers[task_type] = handler
    
    async def start(self, poll_interval: float = 5.0):
        """Start the worker with the given poll interval."""
        self.poll_interval = poll_interval
        self.running = True
        logger.info(f"Worker {self.worker_id} started, polling every {poll_interval}s")
        
        while self.running:
            try:
                # Dequeue jobs for processing
                jobs = await self.job_queue.dequeue(
                    queue_names=None,  # All queues
                    limit=5,  # Process up to 5 jobs per iteration
                    worker_id=self.worker_id
                )
                
                if jobs:
                    logger.info(f"Worker {self.worker_id} dequeued {len(jobs)} jobs")
                    for job in jobs:
                        # Process each job in separate asyncio task
                        task = asyncio.create_task(self._process_job(job))
                        self._current_tasks[job.id] = task
                        task.add_done_callback(lambda t, jid=job.id: self._current_tasks.pop(jid, None))
                else:
                    # No jobs, sleep
                    await asyncio.sleep(self.poll_interval)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)
    
    async def stop(self):
        """Stop the worker gracefully."""
        self.running = False
        logger.info(f"Worker {self.worker_id} stopping...")
        
        # Wait for current tasks to complete with timeout
        if self._current_tasks:
            await asyncio.wait(list(self._current_tasks.values()), timeout=30.0)
    
    async def _process_job(self, job: Job):
        """Process a single job."""
        try:
            # Get associated task
            task = await self.task_manager.get_task(str(job.task_id))
            if not task:
                logger.error(f"Task not found for job {job.id}")
                await self.job_queue.release_lock(str(job.id), self.worker_id)
                return
            
            # Determine task type from metadata
            task_type = task.task_metadata.get('type') if task.task_metadata else 'default'
            handler = self.task_handlers.get(task_type)
            
            if not handler:
                logger.error(f"No handler registered for task type '{task_type}'")
                await self.job_queue.fail(
                    str(job.id),
                    f"No handler registered for task type '{task_type}'",
                    max_attempts_exceeded=True
                )
                return
            
            # Update task status to RUNNING via TaskManager (already done by job status)
            # Execute handler
            try:
                # Prepare arguments from task metadata
                args = task.task_metadata.get('args', []) if task.task_metadata else []
                kwargs = task.task_metadata.get('kwargs', {}) if task.task_metadata else {}
                
                result = await handler(*args, **kwargs)
                
                # Mark job as completed
                await self.job_queue.complete(str(job.id))
                
                # Update task result via TaskManager
                await self.task_manager.set_task_result(str(job.task_id), result)
                await self.task_manager.update_task_status(str(job.task_id), TaskStatus.COMPLETED)
                await self.task_manager.update_task_progress(str(job.task_id), 1.0)
                
                logger.info(f"Job {job.id} completed successfully")
                
            except Exception as e:
                logger.error(f"Handler execution failed for job {job.id}: {e}", exc_info=True)
                
                # Check if max attempts exceeded
                max_attempts_exceeded = job.attempts >= job.max_attempts
                
                await self.job_queue.fail(
                    str(job.id),
                    str(e),
                    error_details={"traceback": str(e.__traceback__)},
                    max_attempts_exceeded=max_attempts_exceeded
                )
                
                # Update task error via TaskManager
                await self.task_manager.set_task_error(str(job.task_id), str(e))
                
                if max_attempts_exceeded:
                    logger.warning(f"Job {job.id} moved to dead letter after {job.attempts} attempts")
        
        except Exception as e:
            logger.error(f"Unexpected error processing job {job.id}: {e}", exc_info=True)
            # Try to release lock
            try:
                await self.job_queue.release_lock(str(job.id), self.worker_id)
            except Exception:
                pass


class EventSystem:
    """Event system for task and job lifecycle events."""
    
    def __init__(self, db_session: AsyncSession = None):
        self.db = db_session
        self._subscribers: Dict[str, List[Callable]] = {}
    
    async def emit(
        self,
        task_id: str,
        event_type: str,
        event_data: Optional[Dict] = None,
        job_id: Optional[str] = None
    ):
        """Emit an event for a task."""
        # Store event in database if db session available
        if self.db:
            from .models import TaskEvent
            event = TaskEvent(
                id=uuid.uuid4(),
                task_id=uuid.UUID(task_id),
                job_id=uuid.UUID(job_id) if job_id else None,
                event_type=event_type,
                event_data=event_data or {},
                created_at=datetime.utcnow()
            )
            self.db.add(event)
            await self.db.commit()
        
        # Notify subscribers
        subscribers = self._subscribers.get(event_type, [])
        for subscriber in subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(task_id, event_type, event_data, job_id)
                else:
                    subscriber(task_id, event_type, event_data, job_id)
            except Exception as e:
                logger.error(f"Event subscriber error: {e}", exc_info=True)
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to events of a specific type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from events."""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)
    
    def clear_subscribers(self, event_type: str = None):
        """Clear subscribers for an event type or all events."""
        if event_type:
            self._subscribers[event_type] = []
        else:
            self._subscribers.clear()


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