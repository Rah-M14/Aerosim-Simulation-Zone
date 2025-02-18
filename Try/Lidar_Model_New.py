import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar
from LiDAR_Fast import *
from torch.utils.data import DataLoader, Dataset
from torch.cuda import amp
import wandb   # <-- For logging

# ---------------------
# Dataset Definition (moved to module-level for pickling)
# ---------------------
class LocalPlannerDataset(Dataset):
    def __init__(self, num_samples, world_limits, binary_img):
        self.num_samples = num_samples
        self.world_limits = world_limits
        self.binary_img = binary_img

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sensor = np.random.uniform(low=[self.world_limits[0][0], self.world_limits[1][0]],
                                   high=[self.world_limits[0][1], self.world_limits[1][1]])
        sensor = validate_agent_position(sensor, self.world_limits, self.binary_img)
        world_theta = np.random.uniform(-np.pi, np.pi)
        _, lidar_dists = get_lidar_points(self.binary_img, sensor, self.world_limits,
                                          num_rays=360, max_range=4.0)
        lidar_dists = np.nan_to_num(lidar_dists, nan=4.0).astype(np.float32)
        sample = {'sensor': sensor.astype(np.float32),
                  'world_theta': np.float32(world_theta),
                  'lidar': lidar_dists}
        return sample

# ---------------------
# Utility/Fallback Functions
# ---------------------

def validate_agent_position(position, world_limits, binary_img, max_offset=1.0):
    """
    Ensure the agent's position lies within world_limits and is not on an obstacle.
    If the position lands on an obstacle, search the nearby area for a free space.
    
    Args:
      position (np.array): [x, y] position in world coordinates.
      world_limits (np.array): Array of shape (2,2): [[min_x, max_x], [min_y, max_y]].
      binary_img (np.array): Binary map where free space is 0 and obstacles are nonzero.
      max_offset (float): Maximum offset to search around the candidate position.
      
    Returns:
      np.array: A valid [x, y] position on free space.
    """
    pos = np.copy(position)
    pos[0] = np.clip(pos[0], world_limits[0][0], world_limits[0][1])
    pos[1] = np.clip(pos[1], world_limits[1][0], world_limits[1][1])
    
    # Convert world coordinates to image coordinates.
    height, width = binary_img.shape
    world_width = world_limits[0][1] - world_limits[0][0]
    world_height = world_limits[1][1] - world_limits[1][0]
    scale_x = width / world_width
    scale_y = height / world_height
    
    # Compute image indices.
    img_x = int(np.rint((pos[0] - world_limits[0][0]) * scale_x))
    img_y = height - 1 - int(np.rint((pos[1] - world_limits[1][0]) * scale_y))
    # Clip indices to ensure they're valid.
    img_x = np.clip(img_x, 0, width - 1)
    img_y = np.clip(img_y, 0, height - 1)
    
    # Free space is assumed to be 0.
    if binary_img[img_y, img_x] == 0:
        return pos
    else:
        num_samples = 11  # sample offsets in each axis.
        for dx in np.linspace(-max_offset, max_offset, num_samples):
            for dy in np.linspace(-max_offset, max_offset, num_samples):
                candidate = pos + np.array([dx, dy])
                candidate[0] = np.clip(candidate[0], world_limits[0][0], world_limits[0][1])
                candidate[1] = np.clip(candidate[1], world_limits[1][0], world_limits[1][1])
                candidate_img_x = int(np.rint((candidate[0] - world_limits[0][0]) * scale_x))
                candidate_img_y = height - 1 - int(np.rint((candidate[1] - world_limits[1][0]) * scale_y))
                # Ensure candidate indices are within image bounds.
                candidate_img_x = np.clip(candidate_img_x, 0, width - 1)
                candidate_img_y = np.clip(candidate_img_y, 0, height - 1)
                if binary_img[candidate_img_y, candidate_img_x] == 0:
                    return candidate
        return pos  # Return clamped position if no free space is found.

def validate_new_position(sensor, predicted_heading, step_size, world_limits, binary_img, max_offset=1.0):
    """
    Compute new position based on sensor position, predicted heading and step size,
    then ensure it is valid.
    """
    new_position = sensor + step_size * np.array([np.cos(predicted_heading), np.sin(predicted_heading)])
    validated = validate_agent_position(new_position, world_limits, binary_img, max_offset)
    return validated

def plot_agent_behavior(binary_img, sensor, world_theta, action, lidar_dists, world_limits, return_fig=False):
    """
    Plot the agent behavior: sensor position, predicted heading, and free (safe) space direction.
    If return_fig is True, return the figure for logging via WandB.
    Otherwise, display the plot for 2 seconds then auto-close.
    """
    img = binary_img.copy()
    img_height, img_width = img.shape
    world_width = world_limits[0][1] - world_limits[0][0]
    world_height = world_limits[1][1] - world_limits[1][0]
    sensor_px = (int((sensor[0] - world_limits[0][0]) / world_width * img_width),
                 int(img_height - (sensor[1] - world_limits[1][0]) / world_height * img_height))
    
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.scatter([sensor_px[0]], [sensor_px[1]], c='red', s=50, label='Sensor')
    
    safe_distance = 1.0
    if isinstance(lidar_dists, torch.Tensor):
        lidar_np = lidar_dists.detach().cpu().numpy()
    else:
        lidar_np = lidar_dists
    ray_angles = torch.linspace(0, 2*np.pi, 361)[:-1]
    free_mask = torch.sigmoid(10.0 * (torch.as_tensor(lidar_np, dtype=torch.float32) - safe_distance))
    ray_angles_np = ray_angles.detach().cpu().numpy()
    free_mask_np = free_mask.detach().cpu().numpy()
    free_sin = np.sum(free_mask_np * np.sin(ray_angles_np)) / (np.sum(free_mask_np) + 1e-6)
    free_cos = np.sum(free_mask_np * np.cos(ray_angles_np)) / (np.sum(free_mask_np) + 1e-6)
    free_direction = np.arctan2(free_sin, free_cos)
    
    arrow_length = 50
    end_x = sensor_px[0] + arrow_length * np.cos(free_direction)
    end_y = sensor_px[1] - arrow_length * np.sin(free_direction)
    ax.arrow(sensor_px[0], sensor_px[1],
             end_x - sensor_px[0], end_y - sensor_px[1],
             color='blue', head_width=5, label='Free Direction')
    
    ax.set_title(f"Sensor: {sensor}, World Theta: {world_theta:.2f}, Action: {action}")
    ax.legend()
    
    if return_fig:
        return fig
    else:
        plt.show(block=False)
        plt.pause(2)
        plt.close('all')

# ---------------------
# Model Definition with BatchNorm
# ---------------------
class LocalPlannerCNN(nn.Module):
    """
    A CNN-based local planner that processes LiDAR data, orientation, and agent position.
    It outputs:
      - L: normalized forward speed (0 to 1)
      - theta_pred_norm: normalized steering adjustment (–1 to 1, to be multiplied by π)
    """
    def __init__(self, num_lidar=360, conv_channels=[32, 64, 128], fc_hidden=128):
        super(LocalPlannerCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv_channels[0],
                               kernel_size=5, padding=2, padding_mode='circular')
        self.bn1 = nn.BatchNorm1d(conv_channels[0])
        self.conv2 = nn.Conv1d(in_channels=conv_channels[0], out_channels=conv_channels[1],
                               kernel_size=5, padding=2, padding_mode='circular')
        self.bn2 = nn.BatchNorm1d(conv_channels[1])
        self.conv3 = nn.Conv1d(in_channels=conv_channels[1], out_channels=conv_channels[2],
                               kernel_size=5, padding=2, padding_mode='circular')
        self.bn3 = nn.BatchNorm1d(conv_channels[2])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # Orientation processing.
        self.fc_theta = nn.Linear(2, 32)  # processes sin and cos of world theta.
        # Sensor (agent) position processing.
        self.fc_pos = nn.Linear(2, 32)
        # Fusion layer.
        self.fc1 = nn.Linear(conv_channels[2] + 32 + 32, fc_hidden)
        self.fc_next = nn.Linear(fc_hidden, fc_hidden // 2)
        # Final heads.
        self.linear_head = nn.Linear(fc_hidden // 2, 1)
        self.angular_head = nn.Linear(fc_hidden // 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, lidar_dists, world_theta, sensor):
        x = lidar_dists.unsqueeze(1)  # (B, 1, 360)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)  # (B, conv_channels[-1])
        
        sin_theta = torch.sin(world_theta).unsqueeze(1)
        cos_theta = torch.cos(world_theta).unsqueeze(1)
        theta_feature = F.relu(self.fc_theta(torch.cat([sin_theta, cos_theta], dim=1)))
        sensor_feature = F.relu(self.fc_pos(sensor))
        
        features = torch.cat([x, theta_feature, sensor_feature], dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc_next(features))
        
        L = self.sigmoid(self.linear_head(features))
        theta_pred_norm = self.tanh(self.angular_head(features))
        return torch.cat([L, theta_pred_norm], dim=1)

# ---------------------
# Training Function with Clear WandB Logging
# ---------------------
def train_local_planner(num_epochs=2000, batch_size=128, lr=1e-3,
                       safe_distance=1.0, alpha=2.0, sigma=0.2,
                       gamma=0.2, temperature=0.5, beta=20.0, delta=2.0,
                       image_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize WandB with all hyperparameters
    wandb.init(project="LocalPlanner_Navigation", config={
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "safe_distance": safe_distance,
        "alpha": alpha,
        "sigma": sigma,
        "gamma": gamma,
        "temperature": temperature,
        "beta": beta,
        "delta": delta
    })
    
    # Create and initialize the model with Xavier/Kaiming initialization
    model = LocalPlannerCNN().to(device)
    def init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(init_weights)
    
    # Enable weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=lr,
                                weight_decay=0.01,  # L2 regularization
                                betas=(0.9, 0.999))  # Adjusted beta parameters
    
    # Use ReduceLROnPlateau to reduce LR when loss plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    scaler = amp.GradScaler()

    world_limits = np.array([[-10, 10], [-8, 8]])
    world_limits_tensor = torch.tensor(world_limits, dtype=torch.float32, device=device)
    
    # Load binary environment image.
    binary_img = create_binary_image(image_path=image_path)
    
    # Create dataset with augmentation
    dataset = LocalPlannerDataset(num_samples=10000, world_limits=world_limits, binary_img=binary_img)
    dataloader = DataLoader(dataset, 
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)  # Improved data loading
    
    # Training loop modifications
    best_loss = float('inf')
    plateau_counter = 0
    patience = 20  # Number of epochs to allow before increasing noise.
    
    model.train()
    pbar = tqdm(range(num_epochs), desc="Training")
    
    for epoch in pbar:
        epoch_loss = 0.0
        epoch_mean_speed = 0.0
        count = 0
        
        for batch in dataloader:
            # Convert batch data efficiently.
            lidar_np = np.array(batch['lidar'])
            lidar_tensor = torch.as_tensor(lidar_np, dtype=torch.float32, device=device)
            world_theta_tensor = torch.as_tensor(np.array(batch['world_theta']), dtype=torch.float32, device=device)
            sensor_tensor = torch.as_tensor(np.array(batch['sensor']), dtype=torch.float32, device=device)
            # Normalize LiDAR data.
            lidar_tensor = (lidar_tensor - lidar_tensor.min()) / (lidar_tensor.max() - lidar_tensor.min() + 1e-8)
            
            with amp.autocast():
                # Get the predicted action.
                actions = model(lidar_tensor, world_theta_tensor, sensor_tensor)
                
                # Increase exploration if plateauing.
                if plateau_counter > patience:
                    noise_factor = min(0.2, 0.1 * (plateau_counter - patience + 1))
                    actions = actions + torch.randn_like(actions) * noise_factor
                
                # Compute L and normalized heading prediction.
                L = torch.clamp(torch.sigmoid(actions[:, 0]), 1e-6, 1.0 - 1e-6)
                theta_pred_norm = torch.clamp(torch.tanh(actions[:, 1]), -1.0 + 1e-6, 1.0 - 1e-6)
                predicted_heading = torch.atan2(
                    torch.sin(world_theta_tensor + theta_pred_norm * np.pi),
                    torch.cos(world_theta_tensor + theta_pred_norm * np.pi)
                )
                
                # Compute predicted new position.
                predicted_new_position = sensor_tensor + (L.unsqueeze(1) *
                                            torch.stack([torch.cos(predicted_heading),
                                                         torch.sin(predicted_heading)], dim=1))
                clamped_x = predicted_new_position[:, 0].clamp(min=world_limits_tensor[0, 0],
                                                               max=world_limits_tensor[0, 1])
                clamped_y = predicted_new_position[:, 1].clamp(min=world_limits_tensor[1, 0],
                                                               max=world_limits_tensor[1, 1])
                predicted_new_position_clamped = torch.stack([clamped_x, clamped_y], dim=1)
                
                # Compute boundary violation.
                violation_x = F.relu(predicted_new_position_clamped[:, 0] - world_limits_tensor[0, 1]) + \
                              F.relu(world_limits_tensor[0, 0] - predicted_new_position_clamped[:, 0])
                violation_y = F.relu(predicted_new_position_clamped[:, 1] - world_limits_tensor[1, 1]) + \
                              F.relu(world_limits_tensor[1, 0] - predicted_new_position_clamped[:, 1])
                boundary_violation = violation_x + violation_y
                boundary_penalty = gamma * F.relu(safe_distance - boundary_violation)
                
                # Compute predicted distance in the forward direction using LiDAR.
                ray_angles = torch.linspace(0, 2*np.pi, 361, device=device)[:-1]
                diff = (ray_angles.unsqueeze(0) - predicted_heading.unsqueeze(1)) % (2*np.pi)
                weights = F.softmax(-0.5 * (diff / sigma)**2, dim=1)
                d_pred = torch.sum(weights * lidar_tensor, dim=1)
                
                best_weights = F.softmax(lidar_tensor / temperature, dim=1)
                best_heading = torch.sum(best_weights * ray_angles.unsqueeze(0), dim=1)
                angle_error = torch.atan2(
                    torch.sin(predicted_heading - best_heading),
                    torch.cos(predicted_heading - best_heading)
                )
                
                min_dist, _ = torch.min(lidar_tensor, dim=1)
                collision_penalty = beta * F.relu(safe_distance - min_dist)
                penalty_forward = F.relu(safe_distance - d_pred)
                
                # NEW REWARD FUNCTION.
                reward = L + 0.8 * min_dist - 1.0 * angle_error**2 - 1.0 * penalty_forward - collision_penalty - boundary_penalty
                mask = torch.isfinite(reward)
                if not mask.all():
                    continue
                loss = -torch.mean(reward[mask])
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # Apply gradient clipping.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            epoch_mean_speed += L.mean().item()
            count += 1
        
        avg_loss = epoch_loss / count if count > 0 else 0.0
        avg_speed = epoch_mean_speed / count if count > 0 else 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        
        pbar.set_description(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Mean Speed: {avg_speed:.4f} | LR: {current_lr:.6f}")
        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "mean_speed": avg_speed,
            "learning_rate": current_lr
        }, step=epoch)
        
        # Plateau detection (loss improvement check).
        if avg_loss > best_loss * 0.995:
            plateau_counter += 1
        else:
            best_loss = avg_loss
            plateau_counter = 0
        
        # Step the scheduler with the average loss.
        scheduler.step(avg_loss)
        
        # Optionally (every 10 epochs) log a sample behavior plot.
        if epoch % 10 == 0:
            sample = next(iter(dataloader))
            sample_sensor = sample['sensor'][0]
            sample_world_theta = sample['world_theta'][0]
            sample_action = actions[0].detach().cpu().numpy()
            print(f"\nEpoch {epoch} - Sample Sensor: {sample_sensor}, World Theta: {sample_world_theta:.2f}, "
                  f"Action: ({L.mean().item():.4f}, {theta_pred_norm.mean().item():.4f})")
            fig = plot_agent_behavior(binary_img, sample_sensor, sample_world_theta,
                                      sample_action, sample['lidar'][0], world_limits, return_fig=True)
            wandb.log({"agent_behavior": wandb.Image(fig)}, step=epoch)
            plt.close(fig)
    
    wandb.finish()
    return model

# ---------------------
# Example usage
# ---------------------
if __name__ == "__main__":
    trained_model = train_local_planner(num_epochs=2000, batch_size=128, lr=5e-3,
                                        safe_distance=1.0, alpha=2.0, sigma=0.2,
                                        gamma=0.2, temperature=0.5, beta=20.0, delta=2.0,
                                        image_path=r"F:\Aerosim-Simulation-Zone\Try\New_WR_World.png")
    trained_model.eval()
    torch.save(trained_model, "batch_local_planner_new.pth")