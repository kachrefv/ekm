from enum import Enum
from sqlalchemy import Column, Integer, String, DateTime, Float, Text, ForeignKey, Boolean, JSON, Index, MetaData, Table
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Import pgvector if available
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    # Define a fallback column type if pgvector is not available
    Vector = JSON  # Use JSON as fallback

# Use a specific MetaData instance to avoid collisions if possible
metadata = MetaData()
Base = declarative_base(metadata=metadata)

# Association table for Gku-AKU many-to-many relationship
gku_aku_association = Table(
    'ekm_gku_akus',
    Base.metadata,
    Column('gku_id', UUID(as_uuid=True), ForeignKey('ekm_gkus.id'), primary_key=True),
    Column('aku_id', UUID(as_uuid=True), ForeignKey('ekm_akus.id'), primary_key=True)
)

class Workspace(Base):
    __tablename__ = 'ekm_workspaces'
    __table_args__ = {'extend_existing': True}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    user_id = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    episodes = relationship("Episode", back_populates="workspace", cascade="all, delete-orphan")
    akus = relationship("AKU", back_populates="workspace", cascade="all, delete-orphan")
    gkus = relationship("GKU", back_populates="workspace", cascade="all, delete-orphan")
    relationships = relationship("AKURelationship", back_populates="workspace", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="workspace", cascade="all, delete-orphan")
    settings = relationship("Setting", back_populates="workspace", cascade="all, delete-orphan")
    sessions = relationship("ChatSession", back_populates="workspace", cascade="all, delete-orphan")
    persona = relationship("Persona", back_populates="workspace", uselist=False, cascade="all, delete-orphan")
    consciousness = relationship("ReflectiveConsciousness", back_populates="workspace", cascade="all, delete-orphan")


class Episode(Base):
    __tablename__ = 'ekm_episodes'
    __table_args__ = {'extend_existing': True}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('ekm_workspaces.id'), nullable=False)
    summary = Column(Text)
    content = Column(Text)
    embedding = Column(Vector if PGVECTOR_AVAILABLE else JSON)
    meta_data = Column(JSON, default=dict) # Using meta_data to avoid conflict with Base.metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    workspace = relationship("Workspace", back_populates="episodes")
    akus = relationship("AKU", back_populates="episode")

class AKU(Base):
    __tablename__ = 'ekm_akus'
    __table_args__ = {'extend_existing': True}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('ekm_workspaces.id'), nullable=False)
    episode_id = Column(UUID(as_uuid=True), ForeignKey('ekm_episodes.id'), nullable=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector if PGVECTOR_AVAILABLE else JSON)
    aku_metadata = Column(JSON, default=dict)
    is_archived = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    workspace = relationship("Workspace", back_populates="akus")
    episode = relationship("Episode", back_populates="akus")
    gkus = relationship("GKU", secondary=gku_aku_association, back_populates="akus")
    source_relationships = relationship("AKURelationship", foreign_keys="AKURelationship.source_aku_id", back_populates="source_aku")
    target_relationships = relationship("AKURelationship", foreign_keys="AKURelationship.target_aku_id", back_populates="target_aku")

class GKU(Base):
    __tablename__ = 'ekm_gkus'
    __table_args__ = {'extend_existing': True}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('ekm_workspaces.id'), nullable=False)
    name = Column(String, nullable=False)  # Name/description of the concept cluster
    description = Column(Text)  # Description of the concept cluster
    centroid_embedding = Column(Vector if PGVECTOR_AVAILABLE else JSON)  # Centroid embedding of the cluster
    pattern_signature = Column(JSON)  # Structural signature of the cluster
    cluster_metadata = Column(JSON, default=dict)  # Additional metadata about the cluster
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    workspace = relationship("Workspace", back_populates="gkus")
    akus = relationship("AKU", secondary=gku_aku_association, back_populates="gkus")

class AKURelationship(Base):
    __tablename__ = 'ekm_aku_relationships'
    __table_args__ = {'extend_existing': True}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('ekm_workspaces.id'), nullable=False)
    source_aku_id = Column(UUID(as_uuid=True), ForeignKey('ekm_akus.id'), nullable=False)
    target_aku_id = Column(UUID(as_uuid=True), ForeignKey('ekm_akus.id'), nullable=False)
    semantic_similarity = Column(Float, default=0.0)
    temporal_proximity = Column(Float, default=0.0)
    causal_weight = Column(Float, default=0.0)
    edge_attributes = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    workspace = relationship("Workspace", back_populates="relationships")
    source_aku = relationship("AKU", foreign_keys=[source_aku_id], back_populates="source_relationships")
    target_aku = relationship("AKU", foreign_keys=[target_aku_id], back_populates="target_relationships")

class TrainingJob(Base):
    __tablename__ = 'ekm_training_jobs'
    __table_args__ = {'extend_existing': True}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('ekm_workspaces.id'), nullable=False)
    status = Column(String, nullable=False, default='pending')
    progress = Column(Integer, default=0)
    message = Column(Text)
    job_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    workspace = relationship("Workspace", back_populates="training_jobs")

class Setting(Base):
    __tablename__ = 'ekm_settings'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('ekm_workspaces.id'), nullable=False)
    key = Column(String, nullable=False)
    value = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    workspace = relationship("Workspace", back_populates="settings")
    __table_args__ = (Index('idx_workspace_key_lib', 'workspace_id', 'key', unique=True), {'extend_existing': True})

class RLState(Base):
    __tablename__ = 'ekm_rl_state'
    __table_args__ = {'extend_existing': True}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String, nullable=False, unique=True, index=True)
    weights = Column(JSON, nullable=False)
    rl_metadata = Column(JSON, default=dict)  # Renamed from 'metadata' (reserved)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class ChatSession(Base):
    __tablename__ = 'ekm_chat_sessions'
    __table_args__ = {'extend_existing': True}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('ekm_workspaces.id'), nullable=False)
    name = Column(String, nullable=False, default="New Chat")
    focus_buffer_state = Column(JSON, default=dict) # Persists FocusBuffer items
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    workspace = relationship("Workspace", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = 'ekm_chat_messages'
    __table_args__ = {'extend_existing': True}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('ekm_chat_sessions.id'), nullable=False)
    role = Column(String, nullable=False) # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    mode_used = Column(String) # 'episodic', 'causal', 'hybrid'
    msg_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("ChatSession", back_populates="messages")

class Persona(Base):
    __tablename__ = 'ekm_personas'
    __table_args__ = {'extend_existing': True}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('ekm_workspaces.id'), nullable=False, unique=True)
    name = Column(String, nullable=False)
    personality = Column(Text)  # e.g., "analytical, curious, helpful"
    voice_style = Column(Text)  # e.g., "professional yet friendly"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    workspace = relationship("Workspace", back_populates="persona")

class ReflectiveConsciousness(Base):
    __tablename__ = 'ekm_consciousness'
    __table_args__ = {'extend_existing': True}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('ekm_workspaces.id'), nullable=False)
    mood = Column(String)  # Current "emotional" state
    thought_summary = Column(Text)  # Summary of recent introspection
    focus_topics = Column(JSON)  # Topics the agent is currently "thinking" about
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    workspace = relationship("Workspace", back_populates="consciousness")


class Task(Base):
    __tablename__ = 'ekm_tasks'
    __table_args__ = {'extend_existing': True}
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('ekm_workspaces.id'), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    status = Column(String, default=TaskStatus.PENDING.value)  # Store as string
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    result = Column(JSON)  # Store result as JSON
    error = Column(Text)   # Store error message if any
    task_metadata = Column(JSON)  # Additional metadata as JSON (renamed from 'metadata')
    version = Column(Integer, default=0, nullable=False)  # For optimistic locking
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    workspace = relationship("Workspace", back_populates="tasks")


class JobStatus(Enum):
    """Status of a job in the queue."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"
    CANCELLED = "cancelled"


class Job(Base):
    """Represents a job in the queue for task execution."""
    __tablename__ = 'ekm_jobs'
    __table_args__ = {'extend_existing': True}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey('ekm_tasks.id'), nullable=False, unique=True)
    status = Column(String, default=JobStatus.PENDING.value)
    queue_name = Column(String, default='default', nullable=False)
    priority = Column(Integer, default=0, nullable=False)  # Lower number = higher priority
    attempts = Column(Integer, default=0, nullable=False)
    max_attempts = Column(Integer, default=3, nullable=False)
    last_attempt_at = Column(DateTime(timezone=True))
    next_attempt_at = Column(DateTime(timezone=True))
    locked_at = Column(DateTime(timezone=True))
    locked_by = Column(String)  # Worker identifier
    scheduled_at = Column(DateTime(timezone=True), server_default=func.now())
    error_message = Column(Text)
    error_details = Column(JSON)  # Detailed error information
    execution_log = Column(JSON, default=list)  # Array of log entries
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    task = relationship("Task", back_populates="job", uselist=False)


class TaskEvent(Base):
    __tablename__ = 'ekm_task_events'
    __table_args__ = {'extend_existing': True}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id = Column(UUID(as_uuid=True), ForeignKey('ekm_tasks.id'), nullable=False)
    job_id = Column(UUID(as_uuid=True), ForeignKey('ekm_jobs.id'), nullable=True)
    event_type = Column(String, nullable=False)  # e.g., 'created', 'started', 'completed', 'failed', 'retry'
    event_data = Column(JSON)  # Additional event data
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    task = relationship("Task", back_populates="events")
    job = relationship("Job")


# Add relationship from Task to Job
Task.job = relationship("Job", back_populates="task", cascade="all, delete-orphan", uselist=False)

# Add relationship from Task to events
Task.events = relationship("TaskEvent", back_populates="task", cascade="all, delete-orphan")


# Add relationship to Workspace
Workspace.tasks = relationship("Task", back_populates="workspace", cascade="all, delete-orphan")
