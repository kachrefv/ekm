import logging
import fastapi.encoders as encoders
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Monkey Patch JSON Encoder for Numpy ---
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
# -------------------------------------------

from .routers import workspaces, chat, tasks, graph
from .core.config import settings

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EKM Research Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(workspaces.router)
app.include_router(chat.router)
app.include_router(tasks.router)
app.include_router(graph.router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
