import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import desi_traffic # Register env

def main():
    env_id = "DesiTraffic-medium-v0"
    
    # Gymnasium requires flattening Dict Observation spaces for standard SB3 models.
    # To keep things simple for this baseline script, we wrap it with FlattenObservation.
    env = gym.make(env_id)
    # OpenEnv requires returning dict representations, but SB3 requires flat arrays for standard MLPs
    env = gym.wrappers.FlattenObservation(env)
    
    # Check the environment works with SB3
    check_env(env)
    
    print(f"Starting training on {env_id}...")
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train for a small number of steps to demonstrate it works
    model.learn(total_timesteps=10000)
    
    model.save("desi_traffic_ppo")
    print("Model saved to desi_traffic_ppo.zip")
    
    # Test with visualization
    print("Testing the trained model with PyGame display...")
    env.close()  # Close the previous training env
    
    # Re-initialize with human rendering and flattening for SB3
    test_env = gym.make(env_id, render_mode="human")
    test_env = gym.wrappers.FlattenObservation(test_env)
    
    obs, info = test_env.reset()
    for _ in range(50):
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = test_env.step(action)
        test_env.render()
        print(f"Action taken: {action}, Reward: {reward}")
        if done or truncated:
            obs, info = test_env.reset()
            
    test_env.close()

if __name__ == "__main__":
    main()
