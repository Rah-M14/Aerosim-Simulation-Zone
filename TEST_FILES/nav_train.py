# import phi.torch
from PIL import Image
import torch
from phi.torch.flow import *

class NavigationNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.control_net = torch.nn.Sequential(
            torch.nn.Linear(6, 16),  # Input: [bot_x, bot_y, goal_x, goal_y, world_theta, relative_theta]
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2), # Output: [L, delta_theta]
            # torch.nn.Tanh()  # Tanh activation for bounded output
            torch.nn.Sigmoid()
        )
        
    def forward(self, current_state):
        return self.control_net(current_state)


def simulate_trajectory(net, initial_pos, goal_pos, max_steps=20):

    current_pos = initial_pos
    theta = math.zeros(initial_pos.shape.non_channel)
    total_loss = 0
    path_length = 0  # Track total distance traveled
    prev_controls = None  # For control smoothness
    net = net.to('cuda')
    # --- New: Store trajectory for final position loss ---
    trajectory = [current_pos]

    eps = 1e-6

    
    for step in range(max_steps):
        delta_pos = goal_pos - current_pos

        # Temporal discount factor ---
        temporal_weight = 0.85 ** (step*-1)
        
        # Calculate relative angle using existing vector components
        relative_theta = math.arctan(delta_pos.vector['y'], divide_by=delta_pos.vector['x']+eps) - theta

        relative_theta = (relative_theta + np.pi) % (2 * np.pi) - np.pi 
        
        # Network input
        net_input = math.stack([
            current_pos.vector['x']/8, 
            current_pos.vector['y']/6,
            goal_pos.vector['x']/8,
            goal_pos.vector['y']/6,
            theta/math.PI,
            relative_theta/math.PI
        ], channel('input_features'))
        
        # Network prediction
        controls = math.native_call(net, net_input)
        L = controls.vector[0]
        delta_theta = controls.vector[1]*math.PI + np.random.uniform(low=0, high=0.5, size=1)[0]*math.PI
        delta_theta = (delta_theta + math.PI) % (2*math.PI) - math.PI

        if prev_controls is not None:
            control_change = math.vec_squared(controls - prev_controls)
            total_loss += 0.25 * math.mean(control_change)
        prev_controls = controls

        # Update orientation with physical constraints
        # theta += math.clip(delta_theta, -math.PI/9, math.PI/9)
        theta += delta_theta

        theta = (theta + np.pi) % (2 * np.pi) - np.pi 
        
        # Calculate movement using existing vector dimension
        delta_x = L * math.cos(theta)
        delta_y = L * math.sin(theta)

        movement = math.stack([delta_x, delta_y], dim=channel(vector='x,y'))

        # --- New: Track path length ---
        path_length += math.vec_length(movement)
        
        # Update position
        new_pos = current_pos + movement
        trajectory.append(new_pos)

        # --- Improved: Discounted position loss ---
        position_loss = temporal_weight * math.vec_length(delta_pos)
        control_loss = 0.1 * (math.abs(delta_theta))
        
        total_loss += math.mean(position_loss  + control_loss) #
        
        current_pos = math.where(math.vec_length(delta_pos) > 0.1, new_pos, current_pos)
    
    final_pos_loss = 10.0 * math.vec_length(trajectory[-1] - goal_pos)
    
    # --- New: Path efficiency penalty ---
    straight_line_dist = math.vec_length(goal_pos - initial_pos)
    efficiency_loss = 0.9 * (path_length / (straight_line_dist + eps))  # Prevent div/0
    
    return total_loss + math.mean(final_pos_loss + efficiency_loss)



def generate_batch(batch_size, min_distance=0.1, max_distance=20):
    batch_dim = batch(samples=batch_size)
    vec_dim = channel(vector='x,y')
    
    # Generate initial positions
    initial_x = math.random_uniform(batch_dim, low=-8, high=8)
    initial_y = math.random_uniform(batch_dim, low=-6,  high=6)
    initial_pos = math.stack([initial_x, initial_y], vec_dim)
    
    # Generate random displacement direction (angles)
    angle = math.random_uniform(batch_dim, low=-math.pi, high=math.pi)
    
    # Generate displacement magnitudes between [min_distance, 2*min_distance]
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


def physics_loss(net, initial_pos, goal_pos):
    return simulate_trajectory(net, initial_pos, goal_pos)


import numpy as np
import cv2

def get_lidar_points(binary_img, current_pos, world_limits, num_rays=360, max_range=4.0):
    """
    Get LiDAR first contact points in world coordinates using vectorized operations
    Args:
        binary_img: Binary image where 0 is obstacle, 1 is free space
        current_pos: (x,y) position of the sensor in world coordinates
        world_limits: Array of [[min_x, max_x], [min_y, max_y]] world boundaries
        num_rays: Number of rays to cast (default 360 for 1-degree resolution)
        max_range: Maximum range of the sensor in world units
    Returns:
        points: Array of shape (360,2) with (x,y) coordinates relative to sensor position,
               zeros for rays that don't hit anything
    """
    height, width = binary_img.shape
    # Calculate transformation factors from world to image
    world_width = world_limits[0][1] - world_limits[0][0]
    world_height = world_limits[1][1] - world_limits[1][0]
    scale_x = width / world_width
    scale_y = height / world_height
    # Convert world position to image coordinates
    img_x = int((current_pos[0] - world_limits[0][0]) * scale_x)
    img_y = height - int((current_pos[1] - world_limits[1][0]) * scale_y)
    # Convert max_range to pixels
    max_range_px = int(max_range * min(scale_x, scale_y))
    # Generate all angles at once
    angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
    # Generate direction vectors for all angles
    directions = np.stack([np.cos(angles), -np.sin(angles)], axis=0)  # Shape: (2, num_rays)
    # Generate all ray lengths at once
    ray_lengths = np.arange(1, max_range_px)  # Shape: (max_range_px-1,)
    # Calculate all possible points for all rays using broadcasting
    ray_points = (ray_lengths[:, np.newaxis, np.newaxis] * directions[np.newaxis, :, :])
    ray_points = np.transpose(ray_points, (0, 2, 1))  # Reshape to (max_range_px-1, num_rays, 2)
    # Add sensor position to all points
    ray_points_x = ray_points[..., 0] + img_x  # Shape: (max_range_px-1, num_rays)
    ray_points_y = ray_points[..., 1] + img_y
    # Convert to integer coordinates
    ray_points_x = ray_points_x.astype(np.int32)
    ray_points_y = ray_points_y.astype(np.int32)
    # Create masks for valid points
    valid_x = (ray_points_x >= 0) & (ray_points_x < width)
    valid_y = (ray_points_y >= 0) & (ray_points_y < height)
    valid_points = valid_x & valid_y
    # Initialize array to store contact points (replacing the contact_points list)
    contact_points = np.zeros((num_rays, 2))
    lidar_dists = np.ones(num_rays) * np.inf
    # Find first contact points for each ray
    for ray_idx in range(num_rays):
        valid_ray_points = valid_points[:, ray_idx]
        if not np.any(valid_ray_points):
            continue
        ray_x = ray_points_x[valid_ray_points, ray_idx]
        ray_y = ray_points_y[valid_ray_points, ray_idx]
        # Check for obstacles along the ray
        ray_values = binary_img[ray_y, ray_x]
        obstacle_indices = np.where(ray_values == 0)[0]
        if len(obstacle_indices) > 0:
            # Get first contact point
            first_contact_idx = obstacle_indices[0]
            px = ray_x[first_contact_idx]
            py = ray_y[first_contact_idx]
            # Convert back to world coordinates
            world_x = (px / scale_x) + world_limits[0][0]
            world_y = world_limits[1][0] + (height - py) / scale_y
            # Calculate relative coordinates
            rel_x = world_x - current_pos[0]
            rel_y = world_y - current_pos[1]
            dists = np.sqrt(rel_x**2 + rel_y**2)
            # Check if within max range
            if dists <= max_range:
                contact_points[ray_idx] = [rel_x, rel_y]
                lidar_dists[ray_idx] = dists
    return contact_points, lidar_dists


import matplotlib.pyplot as plt
import torch

def plot_trajectory(net, initial_pos, goal_pos, max_steps=20):
    """Simulate and plot a single trajectory using PyTorch tensors"""
    with torch.no_grad():
        current_pos = initial_pos.clone()
        goal_pos = goal_pos.clone()
        net = net.to('cuda')
        # net = net.to('cpu')

        print(current_pos, goal_pos)
        
        theta = torch.zeros_like(current_pos[0])
        positions = []
        lidar_pts = []
        
        for stp in range(max_steps):
            delta_pos = goal_pos - current_pos
            relative_theta = torch.atan2(delta_pos[1], delta_pos[0]) - theta

            relative_theta = (relative_theta + np.pi) % (2 * np.pi) - np.pi

            image_path = r"TEST_FILES\New_WR_World.png"
            img = np.array(Image.open(image_path).convert('L'))  # Convert to grayscale
            binary_img = (img > 128).astype(np.uint8)  # Threshold to create a binary map
            binary_img = cv2.resize(binary_img, (0,0), fx=0.25, fy=0.25)
            pts, dists = get_lidar_points(binary_img, [4,4], [(-10, 10),(-8, 8)], num_rays=360, max_range=5.0)
            lidar_pts.append(pts+current_pos.cpu().numpy())
            
            net_input = torch.stack([
                current_pos[0]/8, current_pos[1]/6,
                goal_pos[0]/8, goal_pos[1]/6,
                theta/math.PI, relative_theta/math.PI
            ], dim=-1).unsqueeze(0)
            
            controls = net(net_input)[0]
            L = controls[0]
            delta_theta = controls[1]*math.PI + np.random.uniform(low=0, high=0.25, size=1)[0]*math.PI
            delta_theta = (delta_theta + math.PI) % (2*math.PI) - math.PI

            theta = theta + delta_theta
            theta = (theta + np.pi) % (2 * np.pi) - np.pi 

            print(L, theta, dists.shape)

            # theta += delta_theta
            movement = torch.stack([
                L * torch.cos(theta),
                L * torch.sin(theta)
            ])
            current_pos += movement
            # Append a copy of the numpy array to avoid reference issues
            positions.append(current_pos.cpu().numpy().copy())  # Fixed line
            
            if torch.norm(delta_pos) < 0.1:
                break
        
        positions = np.array(positions)
        # Rest of the plotting code remains the same
        plt.figure(figsize=(8, 6))
        plt.plot(positions[:, 0], positions[:, 1], 'b-o', markersize=4, label='Path')
        for pt in lidar_pts:
            plt.plot(pt[:, 0], pt[:, 1], 'r.', markersize=1)
        plt.scatter(positions[0, 0], positions[0, 1], c='green', s=200, marker='*', label='Start')
        plt.scatter(goal_pos[0].item(), goal_pos[1].item(), c='red', s=200, marker='X', label='Goal')
        plt.title("Navigation Trajectory")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    net = NavigationNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(10000):
        # Generate batch using PhiFlow
        initial_pos, goal_pos = generate_batch(512, min_distance=0.1, max_distance=10.0)

        optimizer.zero_grad()
        output = physics_loss(net, initial_pos, goal_pos)
        loss = output[0] if isinstance(output, tuple) else output
        total_loss = loss.sum  # Sum the loss to get a scalar

        # Skip backward and step if loss is NaN/Inf
        if not torch.isfinite(total_loss):
            print(f"Epoch {epoch}: Loss is NaN/Inf, skipping update.")
            continue

        total_loss.backward()  # Backpropagate

        # Clip gradients to prevent explosion (adjust max_norm as needed)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

        # Check for NaNs in gradients
        has_nan = False
        for p in net.parameters():
            if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
                has_nan = True
                break

        if has_nan:
            print(f"Epoch {epoch}: NaN detected in gradients, skipping step.")
            optimizer.zero_grad()  # Clear gradients to prevent contamination
        else:
            optimizer.step()  # Update parameters if no NaNs

        # Logging and plotting
        if epoch % 500 == 0:
            with torch.no_grad():
                initial_pos, goal_pos = generate_batch(1, 4)
                initial_torch = initial_pos.native("samples,vector").squeeze(0)
                goal_torch = goal_pos.native("samples,vector").squeeze(0)

                loss = physics_loss(net, initial_pos, goal_pos)
                print(f"Epoch {epoch}, Loss: {loss.native().item():.4f}")

                # plot_trajectory(net, initial_torch, goal_torch)

    torch.save(net, 'less_noisy_nav_model.pth')

    net1 = torch.load('less_noisy_nav_model.pth', weights_only=False)
    with torch.no_grad():
        initial_torch = torch.tensor([4.5, -3.2], dtype=torch.float32)
        goal_torch = torch.tensor([2.1, -6.1], dtype=torch.float32)

        loss = physics_loss(net1, initial_pos, goal_pos)
        # print(f"Epoch {epoch}, Loss: {loss.native().item():.4f}")

        plot_trajectory(net1, initial_torch, goal_torch, 60)
