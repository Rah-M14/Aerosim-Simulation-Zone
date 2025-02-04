import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import wandb
import argparse

from tqdm import tqdm
from stable_baselines3 import SAC
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.noise import NormalActionNoise

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

    # Normalize position data
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

    # Add future actions
    df['row_num'] = df.groupby('episode').cumcount()
    
    for step in range(1, 5):
        df[f'future_linear_{step}'] = df.groupby('episode')['action_linear'].shift(-step).fillna(0.0)
        df[f'future_angular_{step}'] = df.groupby('episode')['action_angular'].shift(-step).fillna(0.0)
        
        episode_lengths = df.groupby('episode')['row_num'].transform('max')
        boundary_mask = (episode_lengths - df['row_num']) < step
        df.loc[boundary_mask, f'future_linear_{step}'] = 0.0
        df.loc[boundary_mask, f'future_angular_{step}'] = 0.0
    
    df = df.drop('row_num', axis=1)

    # Define observation and action columns
    obs_cols = (
        ['current_x', 'current_y', 'goal_x', 'goal_y'] +
        [f'waypoint_{i}_x' for i in range(env.chunk_size)] +
        [f'waypoint_{i}_y' for i in range(env.chunk_size)] +
        ['distance_to_goal', 'distance_to_next', 'timestep']
    )
    
    action_cols = ['action_linear', 'action_angular'] + [
        col for i in range(1, 5) 
        for col in [f'future_linear_{i}', f'future_angular_{i}']
    ]

    # Create separate dataframes
    obs_df = df[obs_cols]
    actions_df = df[action_cols]

    # Save to CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    obs_path = f"observations_{timestamp}.csv"
    actions_path = f"actions_{timestamp}.csv"
    
    # print(f"Saving observations to {obs_path}")
    # obs_df.to_csv(obs_path, index=False)
    # print(f"Saving actions to {actions_path}")
    # actions_df.to_csv(actions_path, index=False)
    
    # Convert to numpy arrays for return
    observations = obs_df.values
    actions = actions_df.values
    
    return observations, actions

def batch_action_to_pose(actions, current_poses, dt=1):
    batch_size = actions.shape[0]
    device = actions.device
    
    # Initialize poses tensor to store all 5 future poses
    poses = torch.zeros((batch_size, 5, 2), device=device)
    
    # Initialize current position and orientation
    x = current_poses[:, 0].clone()  # (batch_size,)
    y = current_poses[:, 1].clone()  # (batch_size,)
    theta = torch.zeros(batch_size, device=device)  # Initial orientation
    
    # Calculate poses for each timestep
    for i in range(5):
        linear = actions[:, i*2]      # Linear velocity for timestep i
        angular = actions[:, i*2 + 1] # Angular velocity for timestep i
        
        # Update orientation
        theta = theta + angular * np.pi
        theta = ((theta + np.pi) % (2*np.pi)) - np.pi
        
        # Update position based on current orientation
        x = x + linear * torch.cos(theta)
        y = y + linear * torch.sin(theta)
        
        # Store the new pose
        poses[:, i, 0] = x
        poses[:, i, 1] = y
    
    return poses

def pretrain_policy(
    env,
    policy,
    expert_data,
    device="cuda",
    batch_size=256,
    epochs=20,
    learning_rate=5e-3,
    val_split=0.1,
    early_stopping_patience=20,
):
    """Pretrain policy using behavioral cloning with prediction of future actions and poses"""
    observations, actions = expert_data
    
    # Split data into train and validation sets
    split_idx = int(len(observations) * (1 - val_split))
    train_dataset = ExpertDataset(observations[:split_idx], actions[:split_idx])
    val_dataset = ExpertDataset(observations[split_idx:], actions[split_idx:])

    # Weights for different timesteps and loss components
    time_weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # Decreasing weights for future predictions
    alpha = 0.4  # Weight between linear and angular losses
    beta = 0.3   # Weight between action and pose losses
    dt = 1     # Time step for pose calculation
    
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
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    criterion = nn.MSELoss(reduction='none')
    best_val_loss = float('inf')
    patience_counter = 0

    # torch.save({
    #             # 'epoch': epoch,
    #             'model_state_dict': policy.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': best_val_loss,
    #         }, "best_bc_policy.pth")
    
    policy = policy.to(device)
    # Comment out or remove the torch.compile line
    # policy = torch.compile(policy)  # Remove this line

    for epoch in range(epochs):
        # Training phase
        policy.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_obs, batch_actions in train_pbar:
            batch_obs = batch_obs.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type='cuda'):
                pred_actions = policy(batch_obs)
                
                # Get current poses from observations
                current_poses = torch.stack([
                    batch_obs[:, 0],  # current_x
                    batch_obs[:, 1],  # current_y
                ], dim=1)
                
                # Calculate predicted and target pose sequences
                pred_poses = batch_action_to_pose(pred_actions, current_poses)
                target_poses = batch_action_to_pose(batch_actions, current_poses)
                
                total_loss = 0
                for i in range(5):
                    # Action-based loss for this timestep
                    pred_step_actions = pred_actions[:, i*2:(i+1)*2]
                    target_step_actions = batch_actions[:, i*2:(i+1)*2]
                    
                    loss_actions = criterion(pred_step_actions, target_step_actions)
                    loss_l = loss_actions[:, 0].mean()  # linear velocity loss
                    loss_theta = loss_actions[:, 1].mean()  # angular velocity loss
                    action_loss = alpha * loss_l + (1-alpha) * loss_theta
                    
                    # Pose-based loss for this timestep
                    pose_loss = criterion(pred_poses[:, i], target_poses[:, i]).mean()
                    
                    # Combine losses with weights
                    timestep_loss = time_weights[i] * (beta * action_loss + (1-beta) * pose_loss)
                    total_loss += timestep_loss
                    
                    wandb.log({
                        f"bc_train/t{i}_action_loss": action_loss.item(),
                        f"bc_train/t{i}_pose_loss": pose_loss.item(),
                        f"bc_train/t{i}_total_loss": timestep_loss.item(),
                        f"bc_train/t{i}_action_loss_l": loss_l.item(),
                        f"bc_train/t{i}_action_loss_theta": np.rad2deg(loss_theta.item()),
                        f"bc_train/epoch": epoch
                    })

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            c_loss = total_loss.item()
            train_pbar.set_postfix({'loss': f'{c_loss:.6f}'})
            
            wandb.log({
                "bc_train/total_loss": c_loss,
                "bc_train/learning_rate": optimizer.param_groups[0]['lr']
            })

        scheduler.step()

        # Validation phase
        policy.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")

        for batch_obs, batch_actions in val_pbar:
            batch_obs = batch_obs.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type='cuda'):
                pred_actions = policy(batch_obs)
                
                # Get current poses from observations
                current_poses = torch.stack([
                    batch_obs[:, 0],  # current_x
                    batch_obs[:, 1],  # current_y
                ], dim=1)
                
                # Calculate predicted and target pose sequences
                pred_poses = batch_action_to_pose(pred_actions, current_poses)
                target_poses = batch_action_to_pose(batch_actions, current_poses)
                
                total_loss = 0
                for i in range(5):
                    # Action-based loss for this timestep
                    pred_step_actions = pred_actions[:, i*2:(i+1)*2]
                    target_step_actions = batch_actions[:, i*2:(i+1)*2]
                    
                    loss_actions = criterion(pred_step_actions, target_step_actions)
                    loss_l = loss_actions[:, 0].mean()  # linear velocity loss
                    loss_theta = loss_actions[:, 1].mean()  # angular velocity loss
                    action_loss = alpha * loss_l + (1-alpha) * loss_theta
                    
                    # Pose-based loss for this timestep
                    pose_loss = criterion(pred_poses[:, i], target_poses[:, i]).mean()
                    
                    # Combine losses with weights
                    timestep_loss = time_weights[i] * (beta * action_loss + (1-beta) * pose_loss)
                    total_loss += timestep_loss
                    
                    wandb.log({
                        f"bc_val/t{i}_action_loss": action_loss.item(),
                        f"bc_val/t{i}_pose_loss": pose_loss.item(),
                        f"bc_val/t{i}_total_loss": timestep_loss.item(),
                        f"bc_val/t{i}_action_loss_l": loss_l.item(),
                        f"bc_val/t{i}_action_loss_theta": np.rad2deg(loss_theta.item()),
                        f"bc_val/epoch": epoch
                    })

                v_loss = total_loss.item()
                val_pbar.set_postfix({'loss': f'{v_loss:.6f}'})

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = v_loss / len(val_loader)
        
        wandb.log({
            "bc_val/loss": avg_val_loss,
            "bc_train/avg_loss": avg_train_loss,
            "bc_val/epoch": epoch
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

def main(args):
    algo = args.algo.upper()
    # Initialize wandb
    wandb.init(project="pretraining_BC", name=f"bc_pretraining_{algo}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    
    # Create environment
    env = PathFollowingEnv(
        image_path=r"TEST_FILES\New_WR_World.png",
        algo_run=algo,
        max_episode_steps=1000,
        headless=True,
        enable_wandb=False
    )
    
    # Generate expert demonstrations using RRT*
    csv_file = args.csv
    observations, actions = load_demonstrations(csv_file, env)
    expert_data = (observations, actions)

    # print(F"expert_data: {expert_data}")

    policy_kwargs = {
                "net_arch": dict(
                    pi=[16, 32, 64, 32, 16],
                    qf=[16, 32, 64, 32, 16]
                ),
                "optimizer_class": optim.AdamW,
                "optimizer_kwargs": dict(weight_decay=1e-5)
            }
    
    # Initialize new policy for pretraining
    if algo.upper() == "PPO":
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
            policy_kwargs=policy_kwargs,
        ).policy
    elif algo.upper() == "SAC":
        policy = SAC(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=1000000,
                batch_size=256,
                ent_coef='auto',
                gamma=0.99,
                tau=0.005,
                train_freq=1,
                gradient_steps=1,
                learning_starts=10000,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=1,
            ).policy
    elif algo.upper() == "TD3":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )

        policy = TD3(
                "MlpPolicy",
                env,
                learning_rate=0.001,
                buffer_size=1000000,
                learning_starts=100,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                action_noise=action_noise,
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                device=device,
                verbose=1
            ).policy
        
    for key in policy.state_dict():
        print(key)

    # Pretrain the policy
    pretrained_policy = pretrain_policy(
        env=env,
        policy=policy,
        expert_data=expert_data,
        batch_size=2048,
        epochs=1,
        learning_rate=1e-3,
        device=device
    )
    
    # Save the final pretrained policy
    torch.save({
        'model_state_dict': pretrained_policy.state_dict(),
        # 'final_loss': best_val_loss,
    }, "Mod_PPO_Pretrained.pth")
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="td3", choices=["ppo", "sac", "td3"])
    parser.add_argument("--csv", type=str, default="expert_data.csv")
    args = parser.parse_args()

    main(args)