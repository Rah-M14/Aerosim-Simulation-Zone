import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm
from stable_baselines3 import SAC
from stable_baselines3.common.preprocessing import get_action_dim

from env import PathFollowingEnv

from torch.cuda.amp import autocast, GradScaler
import math
import pandas as pd
from datetime import datetime

class ExpertDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]

def generate_expert_data(env, n_episodes=1000, save_dir="expert_demonstrations"):
    """Generate expert demonstrations and save to CSV files"""
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_file = os.path.join(save_dir, f'expert_data_{timestamp}.csv')
    
    columns = [
        'episode', 'step', 'success',
        'current_x', 'current_y',
        'goal_x', 'goal_y'
    ]
    for i in range(env.chunk_size):
        columns.extend([f'waypoint_{i}_x', f'waypoint_{i}_y'])
    columns.extend(['distance_to_goal', 'distance_to_next', 'timestep',
                   'action_linear', 'action_angular'])
    
    # Write header to CSV
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_file, index=False, header=True)
    
    successful_episodes = 0
    thresh = 0.1
    
    episode_pbar = tqdm(total=n_episodes, desc="Generating episodes", position=0)
    success_pbar = tqdm(total=n_episodes, desc="Successful episodes", position=1)
    
    try:
        for episode in range(n_episodes):
            episode_data = []
            
            obs = env.reset()[0]
            done = False
            truncated = False
            
            path = env.path_manager.get_full_path()
            
            # Skip if path is empty or too short
            if path is None or len(path) < 2:
                episode_pbar.update(1)
                continue
                
            current_pos = env.current_pos
            current_theta = env.agent_theta
            episode_success = False
            
            # Path following progress bar
            path_pbar = tqdm(total=len(path)-1, desc="Following path", 
                           position=2, leave=False)
            
            i = 0
            while i < (len(path) - 1):
                current_waypoint = path[i]
                distance = np.linalg.norm(current_waypoint - current_pos)

                if distance < thresh:
                    i += 1
                    path_pbar.update(1)
                    continue
                
                # Calculate expert action
                dx = current_waypoint[0] - current_pos[0]
                dy = current_waypoint[1] - current_pos[1]
                desired_theta = (np.arctan2(dy, dx) + np.pi) % (2 * np.pi) - np.pi
                
                angle_diff = desired_theta - current_theta
                angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
                
                # Generate expert action
                l = min(distance, 1.0) if abs(angle_diff) < np.pi/4 else 0.0
                theta = np.clip(angle_diff / np.pi, -1.0, 1.0)
                
                action = np.array([l, theta])
                
                # Store step data
                row = {
                    'episode': successful_episodes,
                    'step': env.current_step,
                    'success': False,  # Will update later if successful
                    'current_x': obs[0],
                    'current_y': obs[1],
                    'goal_x': obs[2],
                    'goal_y': obs[3],
                }
                
                # Add waypoints
                for j in range(env.chunk_size):
                    if 4 + j*2 + 1 < len(obs):  # Check if waypoint exists in observation
                        row[f'waypoint_{j}_x'] = obs[4 + j*2]
                        row[f'waypoint_{j}_y'] = obs[4 + j*2 + 1]
                    else:
                        row[f'waypoint_{j}_x'] = obs[-6]  # Use last valid waypoint
                        row[f'waypoint_{j}_y'] = obs[-5]
                
                # Add metrics
                row.update({
                    'distance_to_goal': obs[-3],
                    'distance_to_next': obs[-2],
                    'timestep': obs[-1],
                    'action_linear': action[0],
                    'action_angular': action[1]
                })
                
                episode_data.append(row)
                
                # Update environment
                obs, reward, done, truncated, info = env.step(action)
                
                if done or truncated:
                    episode_success = done
                    break
                    
                current_pos = env.current_pos
                current_theta = env.agent_theta
                path_pbar.update(1)
            
            path_pbar.close()
            
            # If episode completed successfully, save it
            if len(episode_data) > 0:
                # Update success flag for all steps in episode
                for step_data in episode_data:
                    step_data['success'] = True
                
                # Append to CSV file
                df = pd.DataFrame(episode_data)
                df.to_csv(csv_file, mode='a', header=False, index=False)
                
                successful_episodes += 1
                success_pbar.update(1)
                
                # Update progress bar postfix with stats
                success_pbar.set_postfix({
                    'len': len(episode_data),
                    'dist': f"{episode_data[-1]['distance_to_goal']:.3f}"
                })
            
            episode_pbar.update(1)
            episode_pbar.set_postfix({'success_rate': f"{successful_episodes/episode_pbar.n:.2%}"})
            
    finally:
        # Clean up progress bars
        episode_pbar.close()
        success_pbar.close()
    
    return csv_file

def load_demonstrations(csv_file):
    """Load demonstrations from single CSV file"""
    print(f"Loading demonstrations from {csv_file}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter successful episodes only
    df = df[df['success'] == True]
    
    # Extract observations
    obs_cols = (
        ['current_x', 'current_y', 'goal_x', 'goal_y'] +
        [f'waypoint_{i}_x' for i in range(env.chunk_size)] +
        [f'waypoint_{i}_y' for i in range(env.chunk_size)] +
        ['distance_to_goal', 'distance_to_next', 'timestep']
    )
    
    # Extract actions
    action_cols = ['action_linear', 'action_angular']
    
    observations = df[obs_cols].values
    actions = df[action_cols].values
    
    return observations, actions

if __name__ == "__main__":
    env = PathFollowingEnv(
        image_path="standalone_examples/api/omni.isaac.kit/TEST_FILES/New_WR_World.png",
        algo="SAC",
        max_episode_steps=1000,
        headless=True,
        enable_reward_monitor=False,
        enable_wandb=False
    )
    
    # Generate demonstrations
    csv_file = generate_expert_data(env, n_episodes=int(1e6), save_dir="/home/rahm/SIMPLE_LOGS/DATA")
    
    # Load and verify the data
    with tqdm(total=1, desc="Loading and verifying data") as pbar:
        observations, actions = load_demonstrations(csv_file)
        pbar.update(1)
        pbar.set_postfix({
            'obs_shape': observations.shape,
            'act_shape': actions.shape
        })