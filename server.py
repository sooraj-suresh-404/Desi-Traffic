import uvicorn
from fastapi import FastAPI
from openenv.core.env_server.http_server import HTTPEnvServer

from desi_traffic.env import DesiTrafficEnv
from desi_traffic.models import TrafficObservation, TrafficAction

# Environment factory function as required by HTTPEnvServer
def create_env():
    return DesiTrafficEnv(difficulty="medium")

# Initialize the OpenEnv API Wrapper
server = HTTPEnvServer(
    env=create_env,
    action_cls=TrafficAction,
    observation_cls=TrafficObservation,
    max_concurrent_envs=1
)

app = FastAPI()

# Register /reset, /step, /metadata endpoints
server.register_routes(app, mode="simulation")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
