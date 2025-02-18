import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
from phi.torch import *
from phi.torch.flow import *

# Import LiDAR helpers from LiDAR_Fast.py and Lidar_Model_Enc.py
from LiDAR_Fast import get_lidar_points, create_binary_image
# (Make sure that process_safe_angles and its helper group_angle_ranges from Lidar_Model_Enc.py are available in PYTHONPATH.)
from Lidar_Model_Enc import process_safe_angles, group_angle_ranges

# import math

###############################################################################
# 1. NETWORK DEFINITION (INPUT SIZE NOW 12)
###############################################################################

class NavigationNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [start_x, start_y, goal_x, goal_y, world_theta, relative_theta,
        #         lidar_center_1, lidar_center_2, ..., lidar_center_6] (all normalized)
        self.features = torch.nn.Sequential(
            torch.nn.Linear(12, 16),
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
        L = self.linear_head(features).squeeze(-1)      # (batch,)
        delta_theta = self.angular_head(features).squeeze(-1)  # (batch,)
        # Stack outputs into one tensor of shape (batch, 2)
        return torch.stack([L, delta_theta], dim=-1)

###############################################################################
# 2. HELPER: ENCODE LIDAR CENTRES (VECTORIZED GROUPING)
###############################################################################
def encode_lidar_centres(binary_img, sensor_pos, world_limits, threshold, bot_orientation, num_rays=360, max_range=4.0):
    """
    Computes the LiDAR scan and processes the safe angles into centre candidates.
    It then picks the first three infinite and first three finite centres (sorted in
    ascending order of deviation from bot_orientation) and then alternates them to
    produce a fixed 6-element vector.
    """
    # sensor_pos is expected as a numpy array [x, y]
    contact_points, lidar_dists = get_lidar_points(binary_img, sensor_pos, world_limits, 
                                                   num_rays=num_rays, max_range=max_range)
    # Process safe angles (returns lists of centres in degrees)
    infinite_centres, finite_centres, _, _ = process_safe_angles(
        lidar_dists, bot_orientation, threshold, num_rays=num_rays, finite_min_len=1, infinite_min_len=6)
    infinite_centres = np.array(infinite_centres)
    finite_centres = np.array(finite_centres)
    if infinite_centres.size > 0:
        sorted_idx_inf = np.argsort(np.abs(infinite_centres - bot_orientation))
        infinite_centres = infinite_centres[sorted_idx_inf]
    if finite_centres.size > 0:
        sorted_idx_fin = np.argsort(np.abs(finite_centres - bot_orientation))
        finite_centres = finite_centres[sorted_idx_fin]
    # Pick first three entries from each group (if available)
    inf_candidates = infinite_centres[:3] if infinite_centres.size >= 3 else infinite_centres
    fin_candidates = finite_centres[:3] if finite_centres.size >= 3 else finite_centres
    # Alternate the two lists
    combined = []
    for i in range(max(len(inf_candidates), len(fin_candidates))):
        if i < len(inf_candidates):
            combined.append(np.deg2rad(inf_candidates[i]))
        if i < len(fin_candidates):
            combined.append(np.deg2rad(fin_candidates[i]))
    # Pad to ensure six values are returned
    while len(combined) < 6:
        combined.append(bot_orientation)
    return np.array(combined[:6])  # shape (6,)

###############################################################################
# 3. SIMULATION FUNCTION WITH COLLISION AVOIDANCE LOSS
###############################################################################
def simulate_trajectory(net, initial_pos, goal_pos, binary_img, world_limits, threshold, bot_orientation, max_steps=12):
    """
    Simulate a trajectory using the network predictions and augment the loss with a
    collision avoidance term that penalizes steering (theta change) away from the safe
    LiDAR centres.
    """

    # Helper: move a Phi tensor to the proper device by converting its native tensor.
    def to_device(phi_tensor, device):
        print(phi_tensor)
        return math.tensor(phi_tensor.native("samples,vector").to(device))
    
    device = next(net.parameters()).device
    current_pos = to_device(initial_pos, device)
    goal_pos = to_device(goal_pos, device)
    theta = to_device(math.zeros(initial_pos.shape.non_channel), device)  # initial robot orientation

    total_loss = 0
    path_length = 0
    trajectory = [current_pos]
    eps = 1e-6

    # Helper for angular difference (wrap-around)
    def ang_diff(a, b):
        return abs(((a - b + np.pi) % (2 * np.pi)) - np.pi)

    for step in range(max_steps):
        delta_pos = goal_pos - current_pos
        temporal_weight = 0.85 ** (-step)
        # Use index-based tensor access for delta_pos components:
        relative_theta = math.arctan(delta_pos[1], divide_by=delta_pos[0] + eps) - theta
        relative_theta = (relative_theta + np.pi) % (2 * np.pi) - np.pi

        # Convert current_pos (a Phi tensor) into a NumPy vector [x, y] safely:
        sensor_np = np.array([
            current_pos[0].native().cpu().numpy(),
            current_pos[1].native().cpu().numpy()
        ]).squeeze()

        lidar_centres = encode_lidar_centres(binary_img, sensor_np, world_limits, threshold, bot_orientation,
                                               num_rays=360, max_range=4.0)
        # Normalize LiDAR centre angles by PI
        lidar_centres_norm = lidar_centres / np.pi

        # Build the net input (12 features) as a Phi tensor:
        net_input = math.stack([
            current_pos[0] / 10,
            current_pos[1] / 8,
            goal_pos[0] / 10,
            goal_pos[1] / 8,
            theta / math.PI,
            relative_theta / math.PI,
            lidar_centres_norm[0],
            lidar_centres_norm[1],
            lidar_centres_norm[2],
            lidar_centres_norm[3],
            lidar_centres_norm[4],
            lidar_centres_norm[5]
        ], channel("input_features"))
        
        # Move the network input to the same device as the network:
        net_input = to_device(net_input, device)
        # Get network outputs via the native call:
        controls = math.native_call(net, net_input)
        L = controls[0]
        delta_theta = controls[1] * math.PI

        # Collision avoidance: we desire that the new heading (theta + delta_theta)
        # is close to one of the safe LiDAR centre candidates.
        new_heading = theta + delta_theta
        candidate1 = (lidar_centres[0] + lidar_centres[1]) / 2.0
        candidate2 = (lidar_centres[2] + lidar_centres[3]) / 2.0 if len(lidar_centres) >= 4 else candidate1
        candidate3 = (lidar_centres[4] + lidar_centres[5]) / 2.0 if len(lidar_centres) >= 6 else candidate2
        error1 = ang_diff(new_heading, candidate1)
        error2 = ang_diff(new_heading, candidate2)
        error3 = ang_diff(new_heading, candidate3)
        collision_loss = 100 * torch.min(torch.stack([error1, error2, error3])) ** 2
        total_loss += collision_loss

        if step > 0:
            control_change = math.vec_squared(controls - prev_controls)
            total_loss += 0.25 * math.mean(control_change)
        prev_controls = controls

        theta = (theta + delta_theta + np.pi) % (2 * np.pi) - np.pi

        delta_x = L * math.cos(theta)
        delta_y = L * math.sin(theta)
        movement = math.stack([delta_x, delta_y], dim=channel("vector"))
        path_length += math.vec_length(movement)
        new_pos = current_pos + movement
        trajectory.append(new_pos)

        position_loss = temporal_weight * math.vec_length(delta_pos)
        control_loss = 0.1 * math.abs(delta_theta)
        total_loss += math.mean(position_loss + control_loss)
        current_pos = math.where(math.vec_length(delta_pos) > 0.1, new_pos, current_pos)
    
    final_pos_loss = 10.0 * math.vec_length(trajectory[-1] - goal_pos)
    straight_line_dist = math.vec_length(goal_pos - initial_pos)
    efficiency_loss = 0.9 * (path_length / (straight_line_dist + eps))
    
    return total_loss + math.mean(final_pos_loss + efficiency_loss)

###############################################################################
# 4. TRAINING AND UTILITY FUNCTIONS (BATCH, PLOTTING, ETC.)
###############################################################################
def generate_batch(batch_size, min_distance=2.0, max_distance=12):
    batch_dim = batch(samples=batch_size)
    vec_dim = channel("vector")
    initial_x = math.random_uniform(batch_dim, low=-8, high=8)
    initial_y = math.random_uniform(batch_dim, low=-6,  high=6)
    initial_pos = math.stack([initial_x, initial_y], vec_dim)
    angle = math.random_uniform(batch_dim, low=-math.pi, high=math.pi)
    distance = math.random_uniform(batch_dim, low=min_distance, high=max_distance)
    dx = distance * math.cos(angle)
    dy = distance * math.sin(angle)
    goal_x = initial_x + dx
    goal_y = initial_y + dy
    goal_x = math.clip(goal_x, -7.9, 7.9)
    goal_y = math.clip(goal_y, -5.9, 5.9)
    goal_pos = math.stack([goal_x, goal_y], vec_dim)
    return initial_pos, goal_pos

def physics_loss(net, initial_pos, goal_pos, binary_img, world_limits, threshold, bot_orientation):
    return simulate_trajectory(net, initial_pos, goal_pos, binary_img, world_limits, threshold, bot_orientation)

def plot_trajectory(net, initial_pos, goal_pos, binary_img, world_limits, threshold, bot_orientation, max_steps=12):
    """Simulate and plot a single trajectory using PyTorch tensors."""
    with torch.no_grad():
        current_pos = initial_pos.clone()
        goal_pos = goal_pos.clone()
        theta = torch.zeros_like(current_pos[0])
        positions = []
        for stp in range(max_steps):
            delta_pos = goal_pos - current_pos
            relative_theta = torch.atan2(delta_pos[1], delta_pos[0]) - theta
            relative_theta = (relative_theta + np.pi) % (2 * np.pi) - np.pi
            
            # Compute LiDAR centres from current position (as numpy)
            current_pos_np = current_pos.cpu().numpy()
            lidar_centres = encode_lidar_centres(binary_img, current_pos_np, world_limits, threshold, bot_orientation,
                                                  num_rays=360, max_range=4.0)
            lidar_centres_norm = lidar_centres / np.pi
            
            net_input = torch.stack([
                current_pos[0] / 8, current_pos[1] / 6,
                goal_pos[0] / 8, goal_pos[1] / 6,
                theta / math.pi, relative_theta / math.pi,
                torch.tensor(lidar_centres_norm[0], dtype=torch.float32),
                torch.tensor(lidar_centres_norm[1], dtype=torch.float32),
                torch.tensor(lidar_centres_norm[2], dtype=torch.float32),
                torch.tensor(lidar_centres_norm[3], dtype=torch.float32),
                torch.tensor(lidar_centres_norm[4], dtype=torch.float32),
                torch.tensor(lidar_centres_norm[5], dtype=torch.float32)
            ], dim=-1).unsqueeze(0)
            
            controls = net(net_input)[0]
            L = controls[0]
            delta_theta = controls[1] * math.pi
            theta = theta + delta_theta
            theta = (theta + np.pi) % (2 * np.pi) - np.pi
            movement = torch.stack([
                L * torch.cos(theta),
                L * torch.sin(theta)
            ])
            current_pos += movement
            positions.append(current_pos.cpu().numpy().copy())
            if torch.norm(delta_pos) < 0.1:
                break
        
        positions = np.array(positions)
        plt.figure(figsize=(8, 6))
        plt.plot(positions[:, 0], positions[:, 1], 'b-o', markersize=4, label='Path')
        plt.scatter(positions[0, 0], positions[0, 1], c='green', s=200, marker='*', label='Start')
        plt.scatter(positions[-1, 0], positions[-1, 1], c='red', s=200, marker='X', label='Goal')
        plt.title("Navigation Trajectory")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()

###############################################################################
# 5. TRAINING LOOP
###############################################################################
net = NavigationNet()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# Load the binary map image for LiDAR scans (adjust the path as needed)
binary_img = create_binary_image(image_path=r"F:\Aerosim-Simulation-Zone\Try\New_WR_World.png")
world_limits = np.array([[-10, 10], [-8, 8]])
threshold = 3.0
bot_orientation = 0.0

for epoch in range(10000):
    initial_pos, goal_pos = generate_batch(512, min_distance=0.5, max_distance=22.0)
    optimizer.zero_grad()
    loss = physics_loss(net, initial_pos, goal_pos, binary_img, world_limits, threshold, bot_orientation)
    total_loss = loss.sum  # Ensure scalar loss
    if not torch.isfinite(total_loss):
        print(f"Epoch {epoch}: Loss is NaN/Inf, skipping update.")
        continue
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    has_nan = any(p.grad is not None and not torch.all(torch.isfinite(p.grad)) for p in net.parameters())
    if has_nan:
        print(f"Epoch {epoch}: NaN in gradients, skipping step.")
        optimizer.zero_grad()
    else:
        optimizer.step()
    if epoch % 500 == 0:
        with torch.no_grad():
            init_pos, goal_pos = generate_batch(1, 4)
            init_torch = init_pos.native("samples,vector").squeeze(0)
            goal_torch = goal_pos.native("samples,vector").squeeze(0)
            loss_val = physics_loss(net, init_pos, goal_pos, binary_img, world_limits, threshold, bot_orientation)
            print(f"Epoch {epoch}, Loss: {loss_val.native().item():.4f}")
            plot_trajectory(net, init_torch, goal_torch, binary_img, world_limits, threshold, bot_orientation)

with torch.no_grad():
    init_torch = torch.tensor([4.5, -3.2], dtype=torch.float32)
    goal_torch = torch.tensor([2.1, -6.1], dtype=torch.float32)
    loss_val = physics_loss(net, init_torch, goal_torch, binary_img, world_limits, threshold, bot_orientation)
    plot_trajectory(net, init_torch, goal_torch, binary_img, world_limits, threshold, bot_orientation, max_steps=60)

torch.save(net, 'coll_free_model.pth')

###############################################################################
# 6. JUST TESTINF PURPOSES - SIMULATION AND LOSS COMPONENTS DEMO FUNCTION
###############################################################################
def demo_loss_components():
    """
    A sample function that runs one simulation of the navigation agent,
    computes and prints every component of the loss per step, and then prints
    the total loss. This demo is useful for verifying the collision avoidance
    loss (which penalizes deviation from safe LiDAR headings) and the progress loss.
    """
    # Sample initial and goal positions (torch tensors of shape (2,))
    initial_pos = torch.tensor([4.0, -3.0], dtype=torch.float32)
    goal_pos = torch.tensor([2.0, -6.0], dtype=torch.float32)
    
    # Load the binary map image and set simulation parameters.
    binary_img = create_binary_image(image_path=r"F:\Aerosim-Simulation-Zone\Try\New_WR_World.png")
    world_limits = np.array([[-10, 10], [-8, 8]])
    threshold = 3.0
    bot_orientation = 0.0

    # Set the network in evaluation mode for this demo.
    net.eval()

    # Initialize simulation variables.
    current_pos = initial_pos.clone()
    theta = torch.tensor(0.0, dtype=torch.float32)  # initial orientation (radians)
    
    total_progress_loss = 0.0
    total_collision_loss = 0.0
    step_losses = []
    positions = [current_pos.cpu().numpy().copy()]
    
    max_steps = 10
    
    # Helper function to compute angular difference (wrapped in [-pi, pi]).
    def angular_diff(a, b):
        diff = (a - b + np.pi) % (2 * np.pi) - np.pi
        return torch.abs(diff)
    
    for step in range(max_steps):
        delta_pos = goal_pos - current_pos
        distance_to_goal = torch.norm(delta_pos)
        desired_theta = torch.atan2(delta_pos[1], delta_pos[0])
        # Compute relative angle (desired direction relative to current theta)
        rel_theta = desired_theta - theta
        rel_theta = (rel_theta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
        
        # Compute LiDAR centres from the current position.
        current_pos_np = current_pos.detach().cpu().numpy()
        # This returns 6 candidate angles (in degrees)
        lidar_centres = encode_lidar_centres(binary_img, current_pos_np, world_limits, threshold, bot_orientation,
                                              num_rays=360, max_range=4.0)
        # Normalize the lidar centre angles by dividing by pi (for network input scaling)
        lidar_centres_norm = lidar_centres / np.pi
        
        # Create network input: 12-dim vector with normalized current/goal positions and angles.
        net_input = torch.tensor([
            current_pos[0] / 10, 
            current_pos[1] / 8, 
            goal_pos[0] / 10, 
            goal_pos[1] / 8, 
            theta / np.pi, 
            rel_theta / np.pi,
            lidar_centres_norm[0],
            lidar_centres_norm[1],
            lidar_centres_norm[2],
            lidar_centres_norm[3],
            lidar_centres_norm[4],
            lidar_centres_norm[5]
        ], dtype=torch.float32).unsqueeze(0)
        
        # Forward pass through the network.
        controls = net(net_input)[0]
        L = controls[0]
        delta_theta = controls[1] * np.pi
        
        # Predict new heading.
        new_theta = theta + delta_theta
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
        
        # Compute the movement vector.
        movement = torch.stack([L * torch.cos(new_theta), L * torch.sin(new_theta)])
        new_pos = current_pos + movement
        
        # Progress loss: squared distance from the new position to the goal.
        progress_loss = torch.norm(goal_pos - new_pos)**2
        
        # Compute collision avoidance loss.
        # Split the 6 LiDAR centres (in degrees) into 3 candidate groups:
        # Candidate 1: average of first two, Candidate 2: average of next two, Candidate 3: average of last two.
        safe_angles_rad = lidar_centres
        # safe_angles_rad = np.deg2rad(safe_angles_deg)
        candidate1 = np.mean(safe_angles_rad[0:2])
        candidate2 = np.mean(safe_angles_rad[2:4])
        candidate3 = np.mean(safe_angles_rad[4:6])
        safe_candidates = torch.tensor([candidate1, candidate2, candidate3], dtype=torch.float32)
        
        # Compute the minimum angular deviation between the predicted new heading and the candidate safe headings.
        diff1 = angular_diff(new_theta, safe_candidates[0])
        diff2 = angular_diff(new_theta, safe_candidates[1])
        diff3 = angular_diff(new_theta, safe_candidates[2])
        collision_loss = 100*torch.min(torch.stack([diff1, diff2, diff3]))**2
        
        # Total loss for this step.
        step_loss = progress_loss + collision_loss
        
        print(f"Step {step}:")
        print(f"   Predicted L: {L.item():.4f}")
        print(f"   Predicted delta_theta: {delta_theta.item():.4f} rad, ({delta_theta.item()*180/np.pi:.2f}°)")
        print(f"   New theta: {new_theta.item():.4f} rad")
        print(f"   safe theta: {candidate1:.4f} rad")
        print(f"   Progress loss (squared distance): {progress_loss.item():.4f}")
        print(f"   Collision avoidance loss (squared angular diff): {collision_loss.item():.4f}")
        print(f"   Total step loss: {step_loss.item():.4f}\n")
        
        total_progress_loss += progress_loss.item()
        total_collision_loss += collision_loss.item()
        step_losses.append(step_loss.item())
        
        current_pos = new_pos
        theta = new_theta
        positions.append(current_pos.detach().cpu().numpy().copy())
        
        if distance_to_goal < 0.1:
            break
    
    total_loss = sum(step_losses)
    print("=== FINAL LOSSES ===")
    print(f"Total progress loss: {total_progress_loss:.4f}")
    print(f"Total collision avoidance loss: {total_collision_loss:.4f}")
    print(f"Overall total loss: {total_loss:.4f}")
    
    # Plot the trajectory.
    positions = np.array(positions)
    plt.figure(figsize=(8,6))
    plt.plot(positions[:,0], positions[:,1], 'b-o', markersize=4, label="Trajectory")
    plt.scatter(positions[0,0], positions[0,1], c='green', s=200, marker='*', label="Start")
    plt.scatter(goal_pos[0].item(), goal_pos[1].item(), c='red', s=200, marker='X', label="Goal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory and Loss Components Demo")
    plt.legend()
    plt.grid(True)
    plt.show()

# if __name__ == "__main__":
    # For demonstration, try the loss components demo.
    # demo_loss_components()
