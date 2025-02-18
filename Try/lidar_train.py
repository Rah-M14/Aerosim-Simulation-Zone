import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tqdm

import torch
from torch import nn
from phi.torch import *
from phi.flow import *

from LiDAR_Fast import *
from Lidar_Model_Enc import process_safe_angles, create_binary_image

#############################################
# UPDATED NAVIGATION NETWORK (with 12 inputs)
#############################################
class NavigationNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Change input dimension from 6 to 12
        self.features = torch.nn.Sequential(
            torch.nn.Linear(12, 16),  # New input: [bot_x, bot_y, goal_x, goal_y, world_theta, relative_theta,
                                     # lidar0, lidar1, lidar2, lidar3, lidar4, lidar5]
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
        )
        self.linear_head = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()  # L will be in [0,1]
        )
        self.angular_head = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Tanh()  # delta_theta will be in [-1,1]
        )
        
    def forward(self, current_state):
        features = self.features(current_state)
        L = self.linear_head(features).squeeze(-1)  # shape: (batch,)
        delta_theta = self.angular_head(features).squeeze(-1)  # shape: (batch,)
        # Stack outputs into one tensor of shape (batch, 2)
        return torch.stack([L, delta_theta], dim=-1)


####################################################################################
# HELPER FUNCTION: Get LiDAR centres as a (6,) tensor based on current sensor state.
####################################################################################
def get_lidar_centres_tensor(sensor_pos, bot_orientation, binary_img, world_limits, 
                             threshold=3.0, num_rays=360, max_range=4.0):
    """
    Given a sensor position and heading, perform a LiDAR scan and process safe groups.
    This function supports both single (sensor_pos.ndim == 1) and batched (sensor_pos.ndim == 2)
    inputs. It returns a tensor of shape (6,) for a single sample or (batch,6) in batched mode.
    """
    # If sensor_pos is a single sample
    if sensor_pos.ndim == 1:
        # If sensor_pos is not a tensor, convert
        if not torch.is_tensor(sensor_pos):
            sensor_pos = torch.tensor(sensor_pos, dtype=torch.float32)
        sensor_np = sensor_pos.detach().cpu().numpy()
        # Run LiDAR scan (returns contact_points and distances)
        contact_points, lidar_dists = get_lidar_points(binary_img, sensor_np, world_limits,
                                                       num_rays=num_rays, max_range=max_range)
        # Process safe angles. (process_safe_angles expects a scalar orientation.)
        bot_orient_val = bot_orientation.item() if torch.is_tensor(bot_orientation) else bot_orientation
        ic, fc, _, _ = process_safe_angles(lidar_dists, bot_orient_val, threshold,
                                           num_rays=num_rays, finite_min_len=6, infinite_min_len=6, n_centres=3)
        ic = ic[0]  # shape (3,)
        fc = fc[0]  # shape (3,)
        # Interleave the groups: [ic0, fc0, ic1, fc1, ic2, fc2]
        lidar_array = np.empty((6,), dtype=np.float32)
        for i in range(3):
            lidar_array[2*i]     = ic[i]
            lidar_array[2*i + 1] = fc[i]
        # Convert back to torch tensor (and copy sensor device if needed)
        lidar_tensor = torch.tensor(lidar_array, dtype=torch.float32)
        if torch.is_tensor(sensor_pos):
            lidar_tensor = lidar_tensor.to(sensor_pos.device)
        return lidar_tensor
    else:
        # --- Batched version: sensor_pos has shape (batch,2) ---
        batch_size = sensor_pos.shape[0]
        lidar_list = []
        for i in range(batch_size):
            # Process each sample individually.
            single_lidar = get_lidar_centres_tensor(sensor_pos[i],
                                                    bot_orientation[i] if torch.is_tensor(bot_orientation) else bot_orientation[i],
                                                    binary_img, world_limits, threshold, num_rays, max_range)
            lidar_list.append(single_lidar)
        return torch.stack(lidar_list, dim=0)


######################################################################
# Simulate trajectory with collision avoidance loss (for training)
######################################################################
def simulate_trajectory_with_collision(net, initial_pos, goal_pos, binary_img, world_limits, max_steps=22, threshold=3.0):
    """
    Given batched (or single) starting and goal positions, simulate the trajectory.
    Converts inputs (if necessary) to torch tensors and ensures a 2D shape is used
    (i.e. shape (batch,2) for positions). The LiDAR centres (size=6) are appended to the
    six navigation features to form a 12-D input. Additionally, a collision penalty is computed
    based on the difference between the predicted heading (theta+delta_theta)
    and one of the safe candidate headings derived from the LiDAR centres.
    """
    # --- Convert inputs to torch tensors (if needed) and ensure batched dimensions ---
    if not torch.is_tensor(initial_pos):
        initial_pos = torch.tensor(initial_pos, dtype=torch.float32)
    if not torch.is_tensor(goal_pos):
        goal_pos = torch.tensor(goal_pos, dtype=torch.float32)
    # For a single sample, add batch dimension.
    if initial_pos.ndim == 1:
        initial_pos = initial_pos.unsqueeze(0)
    if goal_pos.ndim == 1:
        goal_pos = goal_pos.unsqueeze(0)
    
    device = "cuda"
    current_pos = initial_pos.clone()   # (batch, 2)
    batch_size = current_pos.shape[0]
    theta = torch.zeros(batch_size)  # initial heading (batch,)
    total_loss = torch.tensor(0.0)
    path_length = torch.zeros(batch_size)
    prev_controls = None
    eps = 1e-6

    temporal_weight = 1.0  # initial discount weight

    for step in range(max_steps):
        delta_pos = goal_pos - current_pos  # (batch,2)
        # Compute the angle to the goal for each sample.
        rel_angle = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])  # (batch,)
        relative_theta = rel_angle - theta
        # Wrap angle to [-pi, pi]
        relative_theta = (relative_theta + torch.pi) % (2 * torch.pi) - torch.pi

        # --- Extract LiDAR centres ---
        lidar_centres = get_lidar_centres_tensor(current_pos, relative_theta, binary_img, world_limits, threshold)
        # Ensure lidar_features has the proper batch shape
        if lidar_centres.ndim == 1:
            lidar_features = (lidar_centres / torch.pi).unsqueeze(0)
        else:
            lidar_features = lidar_centres / torch.pi  # (batch,6)

        # --- Build Navigation Features ---
        nav_features = torch.stack([
            current_pos[:, 0] / 10,
            current_pos[:, 1] / 7,
            goal_pos[:, 0]    / 10,
            goal_pos[:, 1]    / 7,
            theta           / torch.pi,
            relative_theta  / torch.pi
        ], dim=1)  # (batch,6)

        # Concatenate the navigation features with the LiDAR centres features (12-D input)
        net_input = torch.cat([nav_features, lidar_features], dim=1).to(device)  # (batch,12)

        # --- Network Prediction ---
        controls = net(net_input)  # (batch,2); controls[:,0] = L, controls[:,1] = delta_theta (scaled)
        L = controls[:, 0].cpu()      # linear velocity (batch,)
        delta_theta = controls[:, 1].cpu() * torch.pi  # scale tanh output to [-pi, pi] (batch,)

        # --- Collision-Avoidance Penalty ---
        desired_heading = theta + delta_theta.detach().cpu()  # (batch,)
        # We assume the lidar centres come in pairs. Create three candidate headings by averaging:
        candidate1 = (lidar_centres[:, 0] + lidar_centres[:, 1]) / 2.0
        candidate2 = (lidar_centres[:, 2] + lidar_centres[:, 3]) / 2.0
        candidate3 = (lidar_centres[:, 4] + lidar_centres[:, 5]) / 2.0

        # Compute angular distances (all in radians)
        def angular_distance(a, b):
            diff = torch.abs(a - b)
            return torch.min(diff, 2 * torch.pi - diff)
        diff1 = angular_distance(desired_heading, candidate1)
        diff2 = angular_distance(desired_heading, candidate2)
        diff3 = angular_distance(desired_heading, candidate3)
        # Choose the candidate giving the smallest difference (optionally weight if desired)
        diff_stack = torch.stack([diff1, diff2, diff3], dim=1)  # (batch,3)
        collision_soft_weights = torch.nn.functional.softmin(diff_stack, dim=1)      # (batch,3)
        collision_penalty = (collision_soft_weights * diff_stack).sum(dim=1)     # (batch)
        # print(f"Coll Soft pen: {diff_stack.mean()}")
        collision_weight = 0.9

        # # Optional: penalty for abrupt control changes (smoothness)
        # if prev_controls is not None:
        #     control_change = (controls - prev_controls).pow(2).sum(dim=1)  # (batch,)
        #     total_loss += 0.25 * control_change.mean()
        # prev_controls = controls.detach()

        # --- Update heading and position ---
        theta = theta + delta_theta
        theta = (theta + torch.pi) % (2 * torch.pi) - torch.pi

        delta_x = L * torch.cos(theta)
        delta_y = L * torch.sin(theta)
        movement = torch.stack([delta_x, delta_y], dim=1)  # (batch,2)
        path_length = path_length + torch.norm(movement, dim=1)
        current_pos = current_pos + movement

        # Per-step losses: distance from goal and control magnitude
        step_position_loss = temporal_weight * torch.norm(delta_pos, dim=1)
        # control_loss = 0.1 * torch.abs(delta_theta)
        # print(f"Step_position_Loss: {step_position_loss.mean()}")
        step_loss = step_position_loss + collision_weight * collision_penalty
        # print(f"Step Loss: {step_loss.mean()}")
        total_loss += step_loss.mean()
        # print(f"Total Loss: {total_loss}")

        temporal_weight *= 0.85

    # Final losses: distance from goal and path efficiency
    final_pos_loss = 10.0 * torch.norm(current_pos - goal_pos, dim=1)
    straight_line_dist = torch.norm(goal_pos - initial_pos, dim=1)
    efficiency_loss = 0.9 * (path_length / (straight_line_dist + eps))
    total_loss += (final_pos_loss.mean() + efficiency_loss.mean())
    return total_loss


#############################################
# Batch generator (unchanged from your code)
#############################################
def generate_batch(batch_size, min_distance=0.5, max_distance=22):
    batch_dim = batch(samples=batch_size)
    vec_dim = channel(vector='x,y')
    
    # Generate initial positions
    initial_x = math.random_uniform(batch_dim, low=-8, high=8)
    initial_y = math.random_uniform(batch_dim, low=-6, high=6)
    initial_pos = math.stack([initial_x, initial_y], vec_dim)
    
    # Generate random displacement direction (angles)
    angle = math.random_uniform(batch_dim, low=-math.pi, high=math.pi)
    
    # Generate displacement magnitudes between [min_distance, max_distance]
    distance = math.random_uniform(batch_dim, low=min_distance, high=max_distance)
    
    # Compute displacement components
    dx = distance * math.cos(angle)
    dy = distance * math.sin(angle)
    
    # Apply displacement to initial positions
    goal_x = (initial_x + dx)
    goal_y = (initial_y + dy)
    
    # Clamp goals to stay within bounds
    goal_x = math.clip(goal_x, -7.9, 7.9)
    goal_y = math.clip(goal_y, -5.9, 5.9)
    
    goal_pos = math.stack([goal_x, goal_y], vec_dim)
    
    return initial_pos, goal_pos


#############################################
# NEW PHYICS + COLLISION LOSS FUNCTION
#############################################
def physics_collision_loss(net, initial_pos, goal_pos, binary_img, world_limits):
    return simulate_trajectory_with_collision(net, initial_pos, goal_pos, binary_img, world_limits)


###############################################################
# A simple plotting function (updated version for a single traj)
###############################################################
def plot_trajectory_with_collision(net, initial_pos, goal_pos, binary_img, world_limits, epoch_no, max_steps=22, threshold=3.0):
    """Simulate and plot a single trajectory using PyTorch tensors (augmented input)."""
    with torch.no_grad():
        # Convert inputs to torch tensors if they are NumPy arrays.
        if not torch.is_tensor(initial_pos):
            initial_pos = torch.tensor(initial_pos, dtype=torch.float32)
        if not torch.is_tensor(goal_pos):
            goal_pos = torch.tensor(goal_pos, dtype=torch.float32)
            
        current_pos = initial_pos.clone()
        theta = torch.zeros(1, device=current_pos.device)
        positions = [current_pos.cpu().numpy()]
        
        for stp in range(max_steps):
            delta_pos = goal_pos - current_pos
            rel_angle = torch.atan2(delta_pos[1:2], delta_pos[0:1])
            relative_theta = rel_angle - theta
            relative_theta = torch.remainder(relative_theta + torch.pi, 2 * torch.pi) - torch.pi

            # Get LiDAR centres from environment. This function handles both single and batched inputs.
            lidar_centres = get_lidar_centres_tensor(current_pos, theta, binary_img, world_limits, threshold=threshold)
            lidar_features = lidar_centres / torch.pi

            nav_features = torch.stack([
                current_pos[0] / 10,
                current_pos[1] / 7,
                goal_pos[0] / 10,
                goal_pos[1] / 7,
                theta[0] / torch.pi,
                relative_theta[0] / torch.pi
            ], dim=-1)
            net_input = torch.cat([nav_features, lidar_features]).unsqueeze(0).to('cuda')
            
            controls = net(net_input)[0]
            L = controls[0].detach().cpu()
            delta_theta = controls[1].detach().cpu() * torch.pi

            theta = theta + delta_theta
            theta = torch.remainder(theta + torch.pi, 2 * torch.pi) - torch.pi 

            movement = torch.stack([L * torch.cos(theta), L * torch.sin(theta)]).squeeze()
            current_pos = current_pos + movement
            positions.append(current_pos.cpu().numpy())
            
            if torch.norm(delta_pos) < 0.1:
                break
        
        positions = np.array(positions)
        plt.figure(figsize=(8, 6))
        plt.plot(positions[:, 0], positions[:, 1], 'b-o', markersize=4, label='Path')
        plt.scatter(positions[0, 0], positions[0, 1], c='green', s=200, marker='*', label='Start')
        plt.scatter(goal_pos[0].item(), goal_pos[1].item(), c='red', s=200, marker='X', label='Goal')
        plt.title("Navigation Trajectory with Collision Avoidance")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show(block=False)
        plt.savefig(os.path.join(r"F:\Aerosim-Simulation-Zone\Try\FIGS", f"{epoch_no}.png"))
        plt.close('all')


#############################################
# TRAINING LOOP
#############################################
device="cuda"
net = NavigationNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# Load the binary image used in LiDAR collision simulation.
binary_img = create_binary_image(image_path=r"F:\Aerosim-Simulation-Zone\Try\New_WR_World.png")
# Define world boundaries (adjust as needed)
world_limits = np.array([[-10, 10], [-8, 8]])
batch_size = 512
for epoch in tqdm.tqdm(range(10000)):
    # Generate a batch using PhiFlow (the original batch generator outputs batched tensors).
    # For our collision simulation we use only one sample at a time.
    initial_pos, goal_pos = generate_batch(batch_size, min_distance=0.5, max_distance=20.0)
    # Convert PhiFlow tensors to PyTorch tensors.
    # (Depending on how PhiFlow math returns its tensors, you may need to call .native())
    initial_pos = initial_pos.native("samples,vector")
    goal_pos = goal_pos.native("samples,vector")

    optimizer.zero_grad()
    loss = physics_collision_loss(net, initial_pos, goal_pos, binary_img, world_limits)
    if not torch.isfinite(loss):
        print(f"Epoch {epoch}: Loss is NaN/Inf, skipping update.")
        continue

    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    
    has_nan = False
    for p in net.parameters():
        if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
            has_nan = True
            break
    if has_nan:
        print(f"Epoch {epoch}: NaN detected in gradients, skipping step.")
        optimizer.zero_grad()
    else:
        optimizer.step()
    
    if epoch % 500 == 0:
        with torch.no_grad():
            # Generate new single-sample test positions.
            initial_pos, goal_pos = generate_batch(1, 4)
            initial_pos = initial_pos.native("samples,vector").squeeze(0)
            goal_pos = goal_pos.native("samples,vector").squeeze(0)
            test_loss = physics_collision_loss(net, initial_pos, goal_pos, binary_img, world_limits)
            print(f"Epoch {epoch}, Loss: {test_loss.item():.4f}")
            plot_trajectory_with_collision(net, initial_pos, goal_pos, binary_img, world_limits, epoch_no=epoch, max_steps=20, threshold=3.0)

# Finally, test on one sample and save the model.
with torch.no_grad():
    initial_torch = torch.tensor([4.5, -3.2], dtype=torch.float32)
    goal_torch = torch.tensor([2.1, -6.1], dtype=torch.float32)
    test_loss = physics_collision_loss(net, initial_torch, goal_torch, binary_img, world_limits)
    plot_trajectory_with_collision(net, initial_torch, goal_torch, binary_img, world_limits, epoch_no=epoch, max_steps=60, threshold=3.0)

torch.save(net, 'coll_nav_model.pth')
