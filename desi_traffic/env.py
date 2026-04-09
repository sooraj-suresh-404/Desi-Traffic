import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from pydantic import ValidationError
from typing import Optional, Tuple, Dict, Any

from .models import TrafficObservation, TrafficAction, TrafficReward, Vector4

# Phase Constants
PHASE_NS_STR = 0
PHASE_EW_STR = 1
PHASE_NS_RGT = 2
PHASE_EW_RGT = 3
PHASE_ALL_RED = 4

class DesiTrafficEnv(gym.Env):
    """
    OpenEnv compliant Gymnasium environment for Indian Traffic Control.
    """
    metadata = {"render_modes": ["console", "human"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, difficulty: str = "easy"):
        super().__init__()
        self.render_mode = render_mode
        self.difficulty = difficulty
        self.renderer = None

        # Rates based on difficulty
        if difficulty == "easy":
            self.arrival_rate_base = 1.0 # vehicles per second
            self.ambulance_prob = 0.01
        elif difficulty == "medium":
            self.arrival_rate_base = 3.0
            self.ambulance_prob = 0.05
        else: # hard
            self.arrival_rate_base = 5.0
            self.ambulance_prob = 0.10

        # Action space: Discrete phase choices (simplified for OpenEnv baseline)
        self.action_space = spaces.Discrete(5)

        # Observation space for classical Gymnasium (normalized between 0 and 1)
        # We will keep it simple: 4 queues, 4 ambulance flags, 1 current phase
        self.observation_space = spaces.Dict({
            "queue_lengths": spaces.Box(low=0, high=1000, shape=(4,), dtype=np.int32),
            "two_wheeler_density": spaces.Box(low=0, high=100, shape=(4,), dtype=np.int32),
            "ambulance_approaching": spaces.MultiBinary(4),
            "current_green_phase": spaces.Discrete(5),
            "phase_timer": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32)
        })

        self.state_data = None
        self.current_step = 0
        self.max_steps = 100

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        
        self.state_data = TrafficObservation(
            queue_lengths=Vector4(north=random.randint(0, 10), south=random.randint(0, 10), east=random.randint(0, 10), west=random.randint(0, 10)),
            two_wheeler_density=Vector4(north=random.randint(20, 80), south=random.randint(20, 80), east=random.randint(20, 80), west=random.randint(20, 80)),
            ambulance_approaching=Vector4(north=0, south=0, east=0, west=0),
            current_green_phase=PHASE_ALL_RED,
            phase_timer=0
        )
        return self.state(), {}

    def state(self) -> Dict:
        """Returns the dictionary representation of the Pydantic observation model."""
        return {
            "queue_lengths": np.array([self.state_data.queue_lengths.north, self.state_data.queue_lengths.south, self.state_data.queue_lengths.east, self.state_data.queue_lengths.west], dtype=np.int32),
            "two_wheeler_density": np.array([self.state_data.two_wheeler_density.north, self.state_data.two_wheeler_density.south, self.state_data.two_wheeler_density.east, self.state_data.two_wheeler_density.west], dtype=np.int32),
            "ambulance_approaching": np.array([self.state_data.ambulance_approaching.north, self.state_data.ambulance_approaching.south, self.state_data.ambulance_approaching.east, self.state_data.ambulance_approaching.west], dtype=np.int8),
            "current_green_phase": self.state_data.current_green_phase,
            "phase_timer": np.array([self.state_data.phase_timer], dtype=np.int32)
        }

    def _get_obs_model(self) -> TrafficObservation:
        return self.state_data

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        action_obj = TrafficAction(next_phase=action, duration=1) # Minimal duration

        if action_obj.next_phase == self.state_data.current_green_phase:
            self.state_data.phase_timer += 1
        else:
            self.state_data.current_green_phase = action_obj.next_phase
            self.state_data.phase_timer = 0

        # Simulate dynamics (1 second step)
        reward_obj = self._simulate_dynamics()
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        return self.state(), reward_obj.overall_score, terminated, truncated, {"pydantic_reward": reward_obj.model_dump(), "pydantic_obs": self.state_data.model_dump()}

    def _simulate_dynamics(self) -> TrafficReward:
        # Stochastic Arrivals
        arrivals = [np.random.poisson(self.arrival_rate_base) for _ in range(4)]
        self.state_data.queue_lengths.north += arrivals[0]
        self.state_data.queue_lengths.south += arrivals[1]
        self.state_data.queue_lengths.east += arrivals[2]
        self.state_data.queue_lengths.west += arrivals[3]

        # Departures based on phase
        cleared = 0
        def clear_queue(queue_val, density):
            base_clear = 3
            # Indian traffic trope: high two-wheeler density clears slightly faster at the start of green light
            modifier = 1.0 + (density / 100.0) * 0.5 
            to_clear = int(np.random.poisson(base_clear * modifier))
            cleared_amt = min(queue_val, to_clear)
            return queue_val - cleared_amt, cleared_amt

        # Update specific queues depending on the green light
        if self.state_data.current_green_phase == PHASE_NS_STR:
            self.state_data.queue_lengths.north, c1 = clear_queue(self.state_data.queue_lengths.north, self.state_data.two_wheeler_density.north)
            self.state_data.queue_lengths.south, c2 = clear_queue(self.state_data.queue_lengths.south, self.state_data.two_wheeler_density.south)
            cleared = c1 + c2
        elif self.state_data.current_green_phase == PHASE_EW_STR:
            self.state_data.queue_lengths.east, c1 = clear_queue(self.state_data.queue_lengths.east, self.state_data.two_wheeler_density.east)
            self.state_data.queue_lengths.west, c2 = clear_queue(self.state_data.queue_lengths.west, self.state_data.two_wheeler_density.west)
            cleared = c1 + c2

        # Stochastic Ambulances
        if random.random() < self.ambulance_prob / 4: self.state_data.ambulance_approaching.north = 1
        if random.random() < self.ambulance_prob / 4: self.state_data.ambulance_approaching.south = 1
        if random.random() < self.ambulance_prob / 4: self.state_data.ambulance_approaching.east = 1
        if random.random() < self.ambulance_prob / 4: self.state_data.ambulance_approaching.west = 1

        # Calculate Rewards
        total_waiting = self.state_data.queue_lengths.north + self.state_data.queue_lengths.south + self.state_data.queue_lengths.east + self.state_data.queue_lengths.west
        wait_penalty = - (total_waiting * 0.1)
        throughput_bonus = cleared * 0.5
        
        amb_penalty = 0.0
        # If an ambulance is waiting on a RED light
        if self.state_data.ambulance_approaching.north and self.state_data.current_green_phase != PHASE_NS_STR: amb_penalty -= 10.0
        elif self.state_data.ambulance_approaching.north and self.state_data.current_green_phase == PHASE_NS_STR: self.state_data.ambulance_approaching.north = 0 # clears immediately
        # Duplicate for others (simplified)
        if self.state_data.ambulance_approaching.south and self.state_data.current_green_phase != PHASE_NS_STR: amb_penalty -= 10.0
        elif self.state_data.ambulance_approaching.south and self.state_data.current_green_phase == PHASE_NS_STR: self.state_data.ambulance_approaching.south = 0
        if self.state_data.ambulance_approaching.east and self.state_data.current_green_phase != PHASE_EW_STR: amb_penalty -= 10.0
        elif self.state_data.ambulance_approaching.east and self.state_data.current_green_phase == PHASE_EW_STR: self.state_data.ambulance_approaching.east = 0
        if self.state_data.ambulance_approaching.west and self.state_data.current_green_phase != PHASE_EW_STR: amb_penalty -= 10.0
        elif self.state_data.ambulance_approaching.west and self.state_data.current_green_phase == PHASE_EW_STR: self.state_data.ambulance_approaching.west = 0

        raw_score = wait_penalty + throughput_bonus + amb_penalty
        
        # Normalize raw score to strictly (0, 1) for validator compliance.
        # Expected raw score range: roughly [-100, +20] per step, so we map to (0, 1).
        epsilon = 1e-6
        bounded_min, bounded_max = -100.0, 20.0
        normalized = (raw_score - bounded_min) / (bounded_max - bounded_min)
        
        # Clamp to open interval with epsilon margins
        if normalized <= 0.0:
            normalized = epsilon
        elif normalized >= 1.0:
            normalized = 1.0 - epsilon
        else:
            normalized = max(epsilon, min(1.0 - epsilon, normalized))
        
        return TrafficReward(total_waiting_time_penalty=wait_penalty, throughput_bonus=throughput_bonus, ambulance_penalty=amb_penalty, overall_score=normalized)

    def render(self):
        if self.render_mode == "console":
            print(f"Step: {self.current_step}")
            print(f"Queues -> N:{self.state_data.queue_lengths.north} S:{self.state_data.queue_lengths.south} E:{self.state_data.queue_lengths.east} W:{self.state_data.queue_lengths.west}")
            print(f"Phase: {self.state_data.current_green_phase} (Timer: {self.state_data.phase_timer})")
            print("-" * 20)
        elif self.render_mode == "human":
            if self.renderer is None:
                from .rendering import DesiTrafficRenderer
                self.renderer = DesiTrafficRenderer()
            # Convert state_data to the generic dict format the renderer expects
            generic_state = {
                'queue_lengths': [self.state_data.queue_lengths.north, self.state_data.queue_lengths.south, self.state_data.queue_lengths.east, self.state_data.queue_lengths.west],
                'two_wheeler_density': [self.state_data.two_wheeler_density.north, self.state_data.two_wheeler_density.south, self.state_data.two_wheeler_density.east, self.state_data.two_wheeler_density.west],
                'ambulance_approaching': [self.state_data.ambulance_approaching.north, self.state_data.ambulance_approaching.south, self.state_data.ambulance_approaching.east, self.state_data.ambulance_approaching.west],
                'current_green_phase': self.state_data.current_green_phase
            }
            self.renderer.render(generic_state)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
