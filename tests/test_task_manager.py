
import pytest
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from ekm.core.models import Base, Task as DBTask, TaskStatus, Workspace
from ekm.core.task_manager import TaskManager, Task

# Setup in-memory DB for testing
# We need to make sure we use a test DB url or mock
TEST_DB_URL = "sqlite:///:memory:"

@pytest.fixture
def db_session():
    engine = create_engine(TEST_DB_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_task_creation_persistence(db_session):
    # 1. Create a dummy workspace
    ws_id = str(uuid.uuid4())
    ws = Workspace(id=uuid.UUID(ws_id), name="Test Workspace", user_id="test_user")
    db_session.add(ws)
    db_session.commit()
    
    # 2. Init TaskManager
    tm = TaskManager(db_session=db_session, workspace_id=ws_id)
    
    # 3. Create Task
    task_id = tm.create_task("Test Task", "Description")
    
    # 4. Verify in DB
    db_task = db_session.query(DBTask).filter_by(id=uuid.UUID(task_id)).first()
    assert db_task is not None
    assert db_task.name == "Test Task"
    assert db_task.status == TaskStatus.PENDING.value
    
    # 5. Verify Retrieval
    retrieved_task = tm.get_task(task_id)
    assert retrieved_task.id == task_id
    assert retrieved_task.name == "Test Task"

def test_task_update(db_session):
    ws_id = str(uuid.uuid4())
    ws = Workspace(id=uuid.UUID(ws_id), name="Test Workspace", user_id="test_user")
    db_session.add(ws)
    db_session.commit()
    
    tm = TaskManager(db_session=db_session, workspace_id=ws_id)
    task_id = tm.create_task("Update Task", "Desc")
    
    # Update Status
    tm.update_task_status(task_id, TaskStatus.RUNNING)
    
    # Verify
    t = tm.get_task(task_id)
    assert t.status == TaskStatus.RUNNING.value
    
    # Update Progress
    tm.update_task_progress(task_id, 0.5)
    t = tm.get_task(task_id)
    assert t.progress == 0.5
    
    # Set Result
    tm.set_task_result(task_id, "Success")
    t = tm.get_task(task_id)
    assert t.result == '"Success"' or t.result == "Success" # JSON serialization check

if __name__ == "__main__":
    # key-man testing if pytest not avail
    try:
        engine = create_engine(TEST_DB_URL)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        sess = Session()
        test_task_creation_persistence(sess)
        test_task_update(sess)
        print("Tests Passed!")
    except Exception as e:
        print(f"Tests Failed: {e}")
        import traceback
        traceback.print_exc()
