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
            torch.nn.Linear(18, 16),  # New input: [bot_x, bot_y, goal_x, goal_y, world_theta, relative_theta,
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
        ic, fc, ib, fb = process_safe_angles(lidar_dists, bot_orient_val, threshold,
                                           num_rays=num_rays, finite_min_len=6, infinite_min_len=6, n_centres=3)
        
        # print(f"ic: {ic.shape}, fc: {fc.shape}")
        lidar_array = np.concatenate([np.array(ic)[:3], np.array(fc)[:3]], axis=1).flatten()
        lidar_c_bounds = np.vstack([np.array(ib)[:3], np.array(fb)[:3]]).flatten()
        # print(f"lidar array: {lidar_array.shape}")
        # Convert back to torch tensor (and copy sensor device if needed)
        lidar_tensor = torch.tensor(lidar_array, dtype=torch.float32)
        lidar_bounds = torch.tensor(lidar_c_bounds, dtype=torch.float32)
        lidar_dists_tensor = torch.tensor(lidar_dists, dtype=torch.float32)
        if torch.is_tensor(sensor_pos):
            lidar_tensor = lidar_tensor.to(sensor_pos.device)
            lidar_bounds = lidar_bounds.to(sensor_pos.device)
            lidar_dists_tensor = lidar_dists_tensor.to(sensor_pos.device)
        return lidar_tensor, lidar_bounds, lidar_dists_tensor
    else:
        # --- Batched version: sensor_pos has shape (batch,2) ---
        batch_size = sensor_pos.shape[0]
        lidar_list = []
        lidar_bounds_list = []
        for i in range(batch_size):
            # Process each sample individually.
            single_lidar, single_bounds, op_dists = get_lidar_centres_tensor(sensor_pos[i],
                                                    bot_orientation[i] if torch.is_tensor(bot_orientation) else bot_orientation[i],
                                                    binary_img, world_limits, threshold, num_rays, max_range)
            lidar_list.append(single_lidar)
            lidar_bounds_list.append(single_bounds)
        return torch.stack(lidar_list, dim=0), torch.stack(lidar_bounds_list, dim=0), op_dists


######################################################################
# Simulate trajectory with collision avoidance loss (for training)
######################################################################
def simulate_trajectory_with_collision(net, initial_pos, goal_pos, binary_img, world_limits, max_steps=22, threshold=3.0):
    """
    Enhanced trajectory simulation with improved collision avoidance using:
    1. Dynamic Window Approach (DWA) safety layer
    2. Trajectory sampling and evaluation
    3. Adaptive velocity scaling based on obstacle proximity
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
    
    # Parameters for collision avoidance
    SAFETY_MARGIN = 0.5  # Minimum distance to obstacles
    MAX_VELOCITY = 1.0   # Maximum linear velocity
    NUM_TRAJECTORIES = 7 # Number of trajectory samples to evaluate
    LOOK_AHEAD_STEPS = 3 # How many steps to simulate ahead for safety checking
    
    for step in range(max_steps):
        delta_pos = goal_pos - current_pos  # (batch,2)
        # Compute the angle to the goal for each sample.
        rel_angle = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])  # (batch,)
        relative_theta = rel_angle - theta
        # Wrap angle to [-pi, pi]
        relative_theta = (relative_theta + torch.pi) % (2 * torch.pi) - torch.pi

        # --- Extract LiDAR centres and bounds ---
        lidar_centres, lidar_bounds, lidar_dists = get_lidar_centres_tensor(current_pos, relative_theta, binary_img, world_limits, threshold)
        lidar_centres, lidar_bounds = (lidar_centres + np.pi) % (2 * np.pi) - np.pi, (lidar_bounds + np.pi) % (2 * np.pi) - np.pi

        # Get raw LiDAR distances - assuming get_lidar_points can be called here
        # (You'll need to modify this to work with your actual LiDAR data structure)
        
        # Compute minimum distance to obstacles for each sample
        min_distances = torch.stack([distances.min() for distances in raw_lidar_distances])
        
        # Ensure lidar_features has the proper batch shape
        if lidar_centres.ndim == 1:
            lidar_features, lidar_bound_features = (lidar_centres / torch.pi).unsqueeze(0), (lidar_bounds / torch.pi).unsqueeze(0)
        else:
            lidar_features, lidar_bound_features = (lidar_centres / torch.pi), (lidar_bounds / torch.pi)  # (batch,6)

        # --- Build Navigation Features ---
        nav_features = torch.stack([
            current_pos[:, 0] / 10,
            current_pos[:, 1] / 7,
            goal_pos[:, 0]    / 10,
            goal_pos[:, 1]    / 7,
            theta           / torch.pi,
            relative_theta  / torch.pi
        ], dim=1)  # (batch,6)

        # Concatenate the navigation features with the LiDAR features
        net_input = torch.cat([nav_features, lidar_bound_features], dim=1).to(device)  # (batch,18)

        # --- Network Prediction ---
        controls = net(net_input)  # (batch,2); controls[:,0] = L, controls[:,1] = delta_theta (scaled)
        
        # --- IMPROVED COLLISION AVOIDANCE ---
        # 1. Adaptive velocity scaling based on proximity to obstacles
        velocity_scale = torch.clamp(min_distances / SAFETY_MARGIN, 0.1, 1.0)
        L = controls[:, 0].cpu() * velocity_scale
        delta_theta = controls[:, 1].cpu() * torch.pi  # scale tanh output to [-pi, pi] (batch,)
        
        # 2. Trajectory sampling and evaluation
        best_controls = []
        for i in range(batch_size):
            # Generate candidate headings around the proposed heading
            candidate_deltas = delta_theta[i] + torch.linspace(-0.5, 0.5, NUM_TRAJECTORIES) * torch.pi/4
            candidate_velocities = L[i] * torch.ones(NUM_TRAJECTORIES)
            
            # For each candidate, simulate trajectory and check for collisions
            best_score = float('-inf')
            best_idx = 0
            
            for j in range(NUM_TRAJECTORIES):
                candidate_theta = theta[i] + candidate_deltas[j]
                candidate_L = candidate_velocities[j]
                
                # Simulate ahead to check for collisions
                test_pos = current_pos[i].clone()
                collision_detected = False
                progress = 0.0
                
                for k in range(LOOK_AHEAD_STEPS):
                    # Move in the candidate direction
                    dx = candidate_L * torch.cos(candidate_theta)
                    dy = candidate_L * torch.sin(candidate_theta)
                    test_pos = test_pos + torch.tensor([dx, dy])
                    
                    # Check if we're getting closer to the goal
                    prev_dist = torch.norm(goal_pos[i] - current_pos[i])
                    new_dist = torch.norm(goal_pos[i] - test_pos)
                    progress += (prev_dist - new_dist)
                    
                    # Check if this position would cause a collision
                    test_pos_np = test_pos.detach().cpu().numpy()
                    _, test_distances = get_lidar_points(binary_img, test_pos_np, world_limits, num_rays=12, max_range=1.0)
                    
                    if min(test_distances) < SAFETY_MARGIN:
                        collision_detected = True
                        break
                
                # Score this candidate trajectory
                # Higher score for trajectories that make more progress toward the goal
                # and don't result in collisions
                collision_penalty = 100.0 if collision_detected else 0.0
                heading_diff = torch.abs(candidate_theta - rel_angle[i])
                heading_diff = min(heading_diff, 2*torch.pi - heading_diff)
                alignment_bonus = 1.0 / (1.0 + heading_diff)
                
                score = progress + alignment_bonus - collision_penalty
                
                if score > best_score:
                    best_score = score
                    best_idx = j
            
            # Use the best controls for this sample
            best_controls.append((candidate_velocities[best_idx], candidate_deltas[best_idx]))
        
        # Update controls with the best ones found
        L_updated = torch.tensor([ctrl[0] for ctrl in best_controls])
        delta_theta_updated = torch.tensor([ctrl[1] for ctrl in best_controls])
        
        # 3. Free space boundary constraints
        num_candidates = lidar_bounds.shape[1] // 2
        lower_bounds = lidar_bounds[:, ::2]  # selects every alternate element starting at index 0
        upper_bounds = lidar_bounds[:, 1::2]  # selects every alternate element starting at index 1
        
        desired_heading = (theta + delta_theta_updated).unsqueeze(1).repeat(1, num_candidates)
        
        # Check if desired heading is within free space bounds
        within_interval = ((desired_heading >= lower_bounds) & (desired_heading <= upper_bounds))
        within_any_interval = within_interval.int().sum(dim=1) > 0
        
        # If not in any interval, adjust to the nearest boundary
        for i in range(batch_size):
            if not within_any_interval[i]:
                heading_i = theta[i] + delta_theta_updated[i]
                
                # Find the nearest boundary
                all_bounds = torch.cat([lower_bounds[i], upper_bounds[i]])
                diffs = torch.abs(all_bounds - heading_i)
                min_idx = torch.argmin(diffs)
                
                # Adjust to the nearest boundary
                delta_theta_updated[i] = all_bounds[min_idx] - theta[i]
        
        # --- Update heading and position ---
        theta = theta + delta_theta_updated
        theta = (theta + torch.pi) % (2 * torch.pi) - torch.pi

        delta_x = L_updated * torch.cos(theta)
        delta_y = L_updated * torch.sin(theta)
        movement = torch.stack([delta_x, delta_y], dim=1)  # (batch,2)
        path_length = path_length + torch.norm(movement, dim=1)
        current_pos = current_pos + movement

        # Per-step losses: distance from goal and control magnitude
        step_position_loss = temporal_weight * torch.norm(delta_pos, dim=1)
        
        # Collision penalty - higher when moving fast near obstacles
        proximity_penalty = 1.0 / (min_distances + eps)
        velocity_penalty = L_updated * proximity_penalty
        step_collision_loss = temporal_weight * velocity_penalty
        
        step_loss = step_position_loss + step_collision_loss
        total_loss += step_loss.mean()

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


def evaluate_model_on_world(net, initial_pos, goal_pos, binary_img, world_limits, max_steps=12, threshold=3.0):
    """
    Evaluate the trained model on the given world image by simulating a trajectory
    from initial_pos to goal_pos and plotting its path overlaid on the binary world image.
    
    Parameters:
      net         : Trained navigation network.
      initial_pos : Start position as a torch tensor (shape: (2,)) or NumPy array.
      goal_pos    : Goal position as a torch tensor (shape: (2,)) or NumPy array.
      binary_img  : The environment's binary image (e.g., produced by create_binary_image()).
      world_limits: NumPy array with shape (2,2): [[xmin, xmax], [ymin, ymax]].
      max_steps   : Maximum steps to simulate.
      threshold   : LiDAR threshold used for extracting safe groups.
      
    This function simulates the trajectory by repeatedly querying the network, and then
    plots the path (with markers for start and goal) atop the world image.
    """
    import matplotlib.pyplot as plt  # Ensure matplotlib is imported here if needed.
    
    with torch.no_grad():
        # Convert inputs to torch tensors if needed.
        if not torch.is_tensor(initial_pos):
            initial_pos = torch.tensor(initial_pos, dtype=torch.float32)
        if not torch.is_tensor(goal_pos):
            goal_pos = torch.tensor(goal_pos, dtype=torch.float32)
        
        # For a single sample, add a batch dimension if needed.
        if initial_pos.ndim == 1:
            initial_pos = initial_pos.unsqueeze(0)
        if goal_pos.ndim == 1:
            goal_pos = goal_pos.unsqueeze(0)
        
        # Begin simulation.
        current_pos = initial_pos.clone()  # (batch, 2)
        batch_size = current_pos.shape[0]
        # We assume a single-sample evaluation so that batch_size==1.
        theta = torch.zeros(batch_size, device=current_pos.device)  # initial heading (batch,)
        positions = [current_pos.cpu().numpy().squeeze()]  # record starting position
        
        for step in range(max_steps):
            delta_pos = goal_pos - current_pos  # (batch, 2)
            rel_angle = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])  # (batch,)
            relative_theta = rel_angle - theta
            relative_theta = (relative_theta + torch.pi) % (2 * torch.pi) - torch.pi
            
            # Extract LiDAR centres (using our batched-friendly version)
            lidar_centres, lidar_bounds, lidar_dists = get_lidar_centres_tensor(current_pos, theta, binary_img, world_limits, threshold=threshold)
            if lidar_centres.ndim == 1:
                lidar_features = (lidar_centres / torch.pi).unsqueeze(0)
            else:
                lidar_features = lidar_centres / torch.pi  # (batch,6)
            
            # Build the 6 navigation features.
            nav_features = torch.stack([
                current_pos[:, 0] / 10,
                current_pos[:, 1] / 7,
                goal_pos[:, 0]    / 10,
                goal_pos[:, 1]    / 7,
                theta           / torch.pi,
                relative_theta  / torch.pi
            ], dim=1)  # (batch,6)
            
            # Concatenate navigation features with LiDAR centres (total 12-D input).
            net_input = torch.cat([nav_features, lidar_features], dim=1)  # (batch,12)
            
            # Get network controls.
            controls = net(net_input)  # (batch,2): controls[:,0]=L, controls[:,1]=delta_theta (scaled)
            L = controls[:, 0]
            delta_theta = controls[:, 1] * torch.pi  # scale tanh output to [-pi, pi]
            
            # Update heading and position.
            theta = theta + delta_theta
            theta = (theta + torch.pi) % (2 * torch.pi) - torch.pi
            
            delta_x = L * torch.cos(theta)
            delta_y = L * torch.sin(theta)
            movement = torch.stack([delta_x, delta_y], dim=1)
            current_pos = current_pos + movement
            
            positions.append(current_pos.cpu().numpy().squeeze())
            
            # Stop if the agent is close enough to the goal.
            if torch.norm(delta_pos, dim=1).item() < 0.1:
                break
        
        # Convert the collected positions to a NumPy array.
        positions = np.array(positions)
        
        # Plot the trajectory overlaying the binary world image.
        plt.figure(figsize=(10, 8))
        # Use extent to align the image to world coordinates.
        extent = (world_limits[0, 0], world_limits[0, 1], world_limits[1, 0], world_limits[1, 1])
        plt.imshow(binary_img, cmap='gray', extent=extent, origin='lower')
        plt.plot(positions[:, 0], positions[:, 1], 'r-o', markersize=5, label='Trajectory')
        plt.scatter(positions[0, 0], positions[0, 1], c='green', s=150, marker='*', label='Start')
        # If goal_pos was unsqueezed, take its first element.
        goal_np = goal_pos.cpu().numpy().squeeze()
        plt.scatter(goal_np[0], goal_np[1], c='blue', s=150, marker='X', label='Goal')
        plt.title("Evaluated Trajectory on World Image")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.show()


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
            lidar_centres, lidar_bounds, lidar_dists = get_lidar_centres_tensor(current_pos, relative_theta, binary_img, world_limits, threshold)
            lidar_centres, lidar_bounds = (lidar_centres + np.pi) % (2 * np.pi) - np.pi, (lidar_bounds + np.pi) % (2 * np.pi) - np.pi

            # Ensure lidar_features has the proper batch shape
            lidar_features, lidar_bound_features = (lidar_centres / torch.pi), (lidar_bounds / torch.pi)  # (batch,6)

            nav_features = torch.stack([
                current_pos[0] / 10,
                current_pos[1] / 7,
                goal_pos[0] / 10,
                goal_pos[1] / 7,
                theta[0] / torch.pi,
                relative_theta[0] / torch.pi
            ], dim=-1)
            net_input = torch.cat([nav_features, lidar_bound_features]).unsqueeze(0).to('cuda')
            
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
        
        img = r"F:\Aerosim-Simulation-Zone\Try\New_WR_World.png"
        positions = np.array(positions)
        plt.figure(figsize=(8, 6))
        plt.imshow(binary_img, extent=[-10, 10, -8, 8], cmap='gray')
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
        # plt.show()
        plt.savefig(os.path.join(r"F:\Aerosim-Simulation-Zone\Try\FIGS_Bound", f"{epoch_no}.png"))
        plt.close('all')


#############################################
# TRAINING LOOP
#############################################
device="cuda"
# model = r"F:\Aerosim-Simulation-Zone\coll_free_nav_model.pth"
# net = torch.load(model, weights_only=False)
net = NavigationNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# Load the binary image used in LiDAR collision simulation.
binary_img = create_binary_image(image_path=r"F:\Aerosim-Simulation-Zone\Try\New_WR_World.png")
# Define world boundaries (adjust as needed)
world_limits = np.array([[-10, 10], [-8, 8]])
batch_size = 512
epochs = 15000
pbar = tqdm.tqdm(range(epochs), desc="Training", dynamic_ncols=True) 

for epoch in pbar:
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
    pbar.set_postfix(loss=loss.item())
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

torch.save(net, 'Bound_based_coll_nav_model.pth')

# # Finally, test on one sample and save the model.
with torch.no_grad():
    initial_torch = torch.tensor([4.8, 6], dtype=torch.float32)
    goal_torch = torch.tensor([-3.1, 0.5], dtype=torch.float32)
    test_loss = physics_collision_loss(net, initial_torch, goal_torch, binary_img, world_limits)
    plot_trajectory_with_collision(net, initial_torch, goal_torch, binary_img, world_limits, epoch_no=1, max_steps=22, threshold=3.0)