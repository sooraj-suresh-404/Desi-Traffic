from gymnasium.envs.registration import register
from .env import DesiTrafficEnv

register(
    id="DesiTraffic-easy-v0",
    entry_point="desi_traffic.env:DesiTrafficEnv",
    kwargs={"difficulty": "easy"}
)

register(
    id="DesiTraffic-medium-v0",
    entry_point="desi_traffic.env:DesiTrafficEnv",
    kwargs={"difficulty": "medium"}
)

register(
    id="DesiTraffic-hard-v0",
    entry_point="desi_traffic.env:DesiTrafficEnv",
    kwargs={"difficulty": "hard"}
)
