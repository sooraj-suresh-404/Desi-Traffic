import os
import json
import gymnasium as gym
from openai import OpenAI
import re
import desi_traffic # This ensures envs are registered
from desi_traffic.grader import grade_episode

# Read environment variables with defaults where required
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def run_inference(env_id: str, task_name: str):
    env = gym.make(env_id)
    obs, info = env.reset()
    
    # Emit START
    print(f"[START] task={task_name} env={env_id} model={MODEL_NAME}")
    
    done = False
    truncated = False
    step_count = 0
    all_rewards = []
    total_penalty = 0

    while not done and not truncated:
        step_count += 1
        
        # Build prompt using current observation
        prompt = f"""
        You are an AI traffic controller managing an Indian city intersection.
        Current Observation: {obs}
        Phase 0: N-S Straight
        Phase 1: E-W Straight
        Phase 2: N-S Right
        Phase 3: E-W Right
        Phase 4: All Red
        Based on queue lengths and ambulance approaching flags, output a SINGLE INTEGER (0, 1, 2, 3, or 4) representing the next phase.
        Return ONLY the integer.
        """
        
        err_msg = "null"
        action = 0 # default safe fall-back
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content.strip()
            # Extract number
            match = re.search(r'\d+', content)
            if match:
                action = int(match.group())
            if action not in [0, 1, 2, 3, 4]:
                action = 4 # default to red on invalid
        except Exception as e:
            err_msg = str(e).replace('\n', ' ')
            action = 4 # fail safe
            
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Tracking
        all_rewards.append(reward)
        pydantic_r = info.get("pydantic_reward", {})
        total_penalty += float(pydantic_r.get("total_waiting_time_penalty", 0)) + float(pydantic_r.get("ambulance_penalty", 0))
        
        # Emit STEP
        done_str = "true" if done else "false"
        print(f"[STEP] step={step_count} action={action} reward={reward:.2f} done={done_str} error={err_msg}")

    # End episode grading
    score = grade_episode(total_penalty)
    success_str = "true" if score >= 0.5 else "false" # Define success threshold
    
    rewards_str = ",".join([f"{r:.2f}" for r in all_rewards])
    print(f"[END] success={success_str} steps={step_count} rewards={rewards_str}")
    env.close()

if __name__ == "__main__":
    tasks = [
        ("DesiTraffic-easy-v0", "desi_traffic_easy"),
        ("DesiTraffic-medium-v0", "desi_traffic_medium"),
        ("DesiTraffic-hard-v0", "desi_traffic_hard")
    ]
    for env_id, task_name in tasks:
        run_inference(env_id, task_name)
