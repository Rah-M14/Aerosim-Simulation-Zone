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

def load_demonstrations(csv_file, env):
    """Load demonstrations from single CSV file"""
    world_max = np.array([8, 6])
    world_limits = np.array([[-8, 8], [-6, 6]])
    world_diag = np.linalg.norm(world_max)
    max_episode_timesteps = 200

    print(f"Loading demonstrations from {csv_file}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter successful episodes only
    df = df[df['success'] == True]

    df['current_x'] = df['current_x'] / world_max[0]
    df['current_y'] = df['current_y'] / world_max[1]
    df['goal_x'] = df['goal_x'] / world_max[0]
    df['goal_y'] = df['goal_y'] / world_max[1]
    for i in range(env.chunk_size):
        df[f'waypoint_{i}_x'] = df[f'waypoint_{i}_x'] / world_max[0]
        df[f'waypoint_{i}_y'] = df[f'waypoint_{i}_y'] / world_max[1]
    df['distance_to_goal'] = df['distance_to_goal'] / world_diag
    df['distance_to_next'] = df['distance_to_next'] / world_diag
    df['timestep'] = df['timestep'] / max_episode_timesteps

    # print(df.head())
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

def pretrain_policy(
    env,
    policy,
    expert_data,
    device="cuda",
    batch_size=512,
    epochs=100,
    learning_rate=3e-4,
    val_split=0.1,
    early_stopping_patience=25,
):
    """Pretrain policy using behavioral cloning with optimized GPU implementation"""
    observations, actions = expert_data
    
    # Split data into train and validation sets
    split_idx = int(len(observations) * (1 - val_split))
    train_dataset = ExpertDataset(observations[:split_idx], actions[:split_idx])
    val_dataset = ExpertDataset(observations[split_idx:], actions[split_idx:])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True
    )

    optimizer = optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=1e-5)
    scaler = GradScaler()  # For mixed precision training
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    
    policy = policy.to(device)
    policy = torch.compile(policy)  # Using torch.compile for speedup
    
    for epoch in range(epochs):
        # Training phase
        policy.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_obs, batch_actions in train_pbar:
            batch_obs = batch_obs.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision training
            with torch.amp.autocast(device_type='cuda'):
                pred_actions = policy(batch_obs)
                loss = criterion(pred_actions[0], batch_actions)
            
            theta_error = criterion(pred_actions[0][:, 1], batch_actions[:, 1])
            # loss += theta_error
            wandb.log({"bc_train/theta_error": np.rad2deg(theta_error.item()*np.pi)})

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            wandb.log({
                "bc_train/loss": loss.item(),
                "bc_train/learning_rate": optimizer.param_groups[0]['lr']
            })

        scheduler.step()

        # Validation phase
        policy.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for batch_obs, batch_actions in val_pbar:
                batch_obs = batch_obs.to(device, non_blocking=True)
                batch_actions = batch_actions.to(device, non_blocking=True)
                
                pred_actions = policy(batch_obs)
                loss = criterion(pred_actions[0], batch_actions)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        wandb.log({
            "bc_val/loss": avg_val_loss,
            "bc_train/avg_loss": avg_train_loss,
            "bc_train/epoch": epoch
        })

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, "best_bc_policy.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    return policy

from stable_baselines3 import SAC, PPO, TD3

def main():
    # Initialize wandb
    wandb.init(project="pretraining_BC", name="bc_pretraining_PPO")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    
    # Create environment
    env = PathFollowingEnv(
        image_path="/home/rahm/.local/share/ov/pkg/isaac-sim-4.2.0/standalone_examples/api/omni.isaac.kit/TEST_FILES/New_WR_World.png",
        algo="PPO",
        max_episode_steps=1000,
        headless=True,
        enable_wandb=False
    )
    
    # Generate expert demonstrations using RRT*
    csv_file = "/home/rahm/SIMPLE_LOGS/expert_data.csv"
    observations, actions = load_demonstrations(csv_file, env)
    expert_data = (observations, actions)

    # print(F"expert_data: {expert_data}")
    
    # Initialize new policy for pretraining
    policy = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        verbose=1,
        n_steps=2048,
        n_epochs=10,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        device=device,
        policy_kwargs={
            "net_arch": dict(
                pi=[256, 512, 512, 256],
                qf=[256, 512, 512, 256]
            ),
            "optimizer_class": optim.AdamW,
            "optimizer_kwargs": dict(weight_decay=1e-5)
        }
    ).policy
    
    # Pretrain the policy
    pretrained_policy = pretrain_policy(
        env=env,
        policy=policy,
        expert_data=expert_data,
        batch_size=1024,
        epochs=100,
        learning_rate=3e-4,
        device=device
    )
    
    # Save the final pretrained policy
    torch.save({
        'model_state_dict': pretrained_policy.state_dict(),
        # 'final_loss': best_val_loss,
    }, "Nnew2_final_bc_policy.pth")
    
    wandb.finish()

if __name__ == "__main__":
    main()