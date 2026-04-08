import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional

from desi_traffic.env import DesiTrafficEnv
from desi_traffic.models import TrafficAction, TrafficObservation

app = FastAPI()

# Maintain a global environment instance since DesiTrafficEnv is stateful
global_env = DesiTrafficEnv(difficulty="medium")

@app.post("/reset")
async def reset_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
        
    seed = body.get("seed") if isinstance(body, dict) else None
    
    # Run reset (which creates the internal state_data numpy arrays)
    global_env.reset(seed=seed)
    
    # Use model_dump to extract pure Python primitives (no numpy arrays!)
    clean_obs = global_env.state_data.model_dump()
    
    return {
        "observation": clean_obs,
        "reward": None,
        "done": False,
        "metadata": {}
    }

@app.post("/step")
async def step_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
        
    if not isinstance(body, dict):
        body = {}

    action_data = body.get("action", {})
    if not isinstance(action_data, dict):
        action_data = {}

    try:
        # Strict mapping through pydantic
        action_obj = TrafficAction(**action_data)
        phase_choice = action_obj.next_phase
    except Exception:
        # Fallback
        phase_choice = action_data.get("next_phase", 0)

    # Force cast to integer to prevent `None` ValidationErrors tumbling up
    try:
        phase_choice = int(phase_choice)
    except (TypeError, ValueError):
        phase_choice = 0

    _, reward, terminated, truncated, info = global_env.step(phase_choice)
    
    # Use model_dump to extract pure Python primitives (no numpy arrays!)
    clean_obs = global_env.state_data.model_dump()
    
    return {
        "observation": clean_obs,
        "reward": float(reward),
        "done": bool(terminated or truncated),
        "metadata": info
    }

@app.get("/schema")
def schema_endpoint():
    return {
        "action": TrafficAction.model_json_schema(),
        "observation": TrafficObservation.model_json_schema(),
        "state": {}
    }

@app.get("/health")
def health_endpoint():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
