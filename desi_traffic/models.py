from pydantic import BaseModel, Field
from typing import List, Dict

class Vector4(BaseModel):
    north: int = Field(default=0, description="Value for North approach")
    south: int = Field(default=0, description="Value for South approach")
    east: int = Field(default=0, description="Value for East approach")
    west: int = Field(default=0, description="Value for West approach")

class TrafficObservation(BaseModel):
    queue_lengths: Vector4 = Field(..., description="Number of vehicles waiting at each approach.")
    two_wheeler_density: Vector4 = Field(..., description="Estimated density of 2-wheelers at each approach (0-100).")
    ambulance_approaching: Vector4 = Field(..., description="Boolean 1 or 0 if an ambulance is approaching.")
    current_green_phase: int = Field(..., description="0=North-South, 1=East-West, 2=North-South Right, 3=East-West Right, 4=All Red")
    phase_timer: int = Field(..., description="Time in seconds that the current phase has been active.")

class TrafficAction(BaseModel):
    next_phase: int = Field(..., description="The next phase to switch to. 0=North-South, 1=East-West, 2=North-South Right, 3=East-West Right, 4=All Red")
    duration: int = Field(default=10, description="Duration in seconds to hold this phase (must be > 5).")

class TrafficReward(BaseModel):
    total_waiting_time_penalty: float = Field(..., description="Negative reward for total waiting time.")
    throughput_bonus: float = Field(..., description="Positive reward for clearing vehicles.")
    ambulance_penalty: float = Field(..., description="Massive negative reward if ambulance is stopped.")
    overall_score: float = Field(..., description="Sum of all reward components.")
