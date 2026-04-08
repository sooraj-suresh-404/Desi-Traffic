import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional

from desi_traffic.env import DesiTrafficEnv
from desi_traffic.models import TrafficAction, TrafficObservation

app = FastAPI()

# Maintain a global environment instance since DesiTrafficEnv is stateful
global_env = DesiTrafficEnv(difficulty="medium")

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None

class StepRequest(BaseModel):
    action: Dict[str, Any]

@app.post("/reset")
def reset_endpoint(request: ResetRequest):
    obs, info = global_env.reset(seed=request.seed)
    return {
        "observation": obs,
        "reward": None,
        "done": False,
        "metadata": info
    }

@app.post("/step")
def step_endpoint(request: StepRequest):
    # OpenEnv evaluator usually sends dynamic JSON matching the Action Schema
    try:
        # Strict mapping through pydantic
        action_data = TrafficAction(**request.action)
        phase_choice = action_data.next_phase
    except Exception:
        # Fallback if action is slightly malformed or purely an integer index
        phase_choice = request.action.get("next_phase", 0)

    obs, reward, terminated, truncated, info = global_env.step(phase_choice)
    return {
        "observation": obs,
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
