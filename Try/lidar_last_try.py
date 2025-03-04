import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from phi.torch import *
from phi.flow import *

from LiDAR_Fast import *
from Lidar_Model_Enc import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#############################################
# UPDATED NAVIGATION NETWORK (with 12 inputs)
#############################################
class NavigationNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Change input dimension from 6 to 12
        self.features = torch.nn.Sequential(
            torch.nn.Linear(6, 16),  # New input: [bot_x, bot_y, goal_x, goal_y, world_theta, relative_theta]
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU()
        )
        self.linear_head = torch.nn.Sequential(
            # torch.nn.Linear(64, 32),
            # torch.nn.ReLU(),
            # torch.nn.Linear(32, 16),
            # torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()  # L will be in [0,1]
        )
        self.angular_head = torch.nn.Sequential(
            # torch.nn.Linear(64, 32),
            # torch.nn.ReLU(),
            # torch.nn.Linear(32, 16),
            # torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()  # delta_theta will be in [0,1]
        )
        # self.distance_temp = nn.Parameter(torch.tensor(2.0))
        self.distance_temp = torch.tensor(2)

    def forward(self, state, safe_angle_mask):
        """
        state: [batch_size, 6] (input features)
        safe_angle_mask: [batch_size, angular_res] (1=safe, 0=unsafe)
        """
        # Base policy output
        # print(f"state shape: {state.shape}")
        raw_output = self.features(state)

        L = self.linear_head(raw_output)  # [0,1] magnitude
        theta_raw = (self.angular_head(raw_output) * (2 * torch.pi)) % (2 * torch.pi)  # [0, 2π]

        print(f"L: {L.shape}, theta_raw: {theta_raw.shape}")

        # print(f"MODEL: Theta_Raw: {torch.rad2deg(theta_raw)}")
        # Convert to angular bins [0, angular_res)
        # theta_deg = torch.rad2deg(theta_raw + torch.pi) % 360
        theta_deg = torch.rad2deg(theta_raw)
        theta_bin = theta_deg.long()
        # --- Differentiable Nearest Safe Angle Selection ---
        # 1. Create distance matrix
        all_bins = torch.arange(360, device=state.device)
        bin_distances = torch.abs(all_bins.float() - theta_bin)
        # print(f"Bin distances: {bin_distances}")
        print(f"all_bins: {all_bins.shape}, bin_distances: {bin_distances.shape}, theta_bin: {theta_bin.shape}")
        
        # 2. Apply safety mask and distance weighting
        safe_weights = F.softmax(-bin_distances / self.distance_temp, dim=-1)
        # print(f"Safe weights: {safe_weights}")
        # plt.plot(safe_weights.detach().cpu().numpy().flatten(), label='weights')
        # plt.show()
        safe_weights = safe_weights * safe_angle_mask  # Zero out unsafe
        # print(f"safe weights : {safe_weights.shape}")

        # 3. Differentiable nearest selection (Gumbel-softmax)
        safe_probs = F.gumbel_softmax(safe_weights.log(), tau=0.5, hard=True)
        # print(f"safe Probs : {safe_probs.shape}")
        print(f"safe_weights: {safe_weights.shape}, safe_probs: {safe_probs.shape}")

        # 4. Get nearest safe angle
        nearest_bin = torch.argmax(safe_probs, dim=-1)
        theta_safe_deg = (nearest_bin.float())
        theta_safe = torch.deg2rad(theta_safe_deg)
        # print(f"theta safe: {theta_safe.shape}")
        # print(f"theta raw: {theta_raw.shape}")
        print(f"nearest_bin: {nearest_bin.shape}, theta_safe: {theta_safe.shape}")
    
        # 5. Blend with raw angle using distance confidence
        confidence = 1 / (1 + bin_distances.gather(-1, nearest_bin.unsqueeze(-1)).squeeze())
        final_theta = confidence * theta_raw.squeeze() + (1 - confidence) * theta_safe
        # print(f"MODEL: Theta_Final: {torch.rad2deg(final_theta)}")
        # print(f"MODEL: Difference: {torch.rad2deg(final_theta - theta_raw)}")
        print(f"confidence: {confidence.shape}, final_theta: {final_theta.shape}")


        # plt.plot(bin_distances[0].cpu().numpy().flatten())
        # plt.plot(safe_weights.detach().cpu().numpy().flatten(), label='weights')
        # plt.plot(safe_probs.detach().cpu().numpy().flatten(), label='probs')
        # plt.plot(safe_weights.detach().cpu().numpy().flatten(), label='weights')
        # plt.show()

        # print(f"confidence: {confidence.shape}")
        # print(f"L final: {L.shape}")
        # print(f"theta final: {final_theta.shape}")



        return torch.stack([L.squeeze(-1), final_theta], dim=-1), torch.stack([L.squeeze(-1), theta_raw.squeeze(-1)], dim=-1) 
        
    # def forward(self, current_state):
    #     features = self.features(current_state)
    #     L = self.linear_head(features).squeeze(-1)  # shape: (batch,)
    #     delta_theta = self.angular_head(features).squeeze(-1)  # shape: (batch,)
    #     # Stack outputs into one tensor of shape (batch, 2)
    #     return torch.stack([L, delta_theta], dim=-1)


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
    # if sensor_pos.ndim == 1:
    # If sensor_pos is not a tensor, convert
    # if not torch.is_tensor(sensor_pos):
    #     sensor_pos = torch.tensor(sensor_pos, dtype=torch.float32)
    sensor_np = sensor_pos.detach().cpu().numpy()
    # Run LiDAR scan (returns contact_points and distances)
    contact_points, lidar_dists = get_lidar_points(binary_img, sensor_np, world_limits,
                                                    num_rays=num_rays, max_range=max_range)
    
    mask = get_safe_mask(lidar_dists, threshold=1.2)    # safe_mask = process_mask(mask, 6)
    lidar_tensor = None

    # # Process safe angles. (process_safe_angles expects a scalar orientation.)
    # bot_orient_val = bot_orientation.item() if torch.is_tensor(bot_orientation) else bot_orientation
    # ic, fc, ib, fb = process_safe_angles(lidar_dists, bot_orient_val, threshold,
    #                                     num_rays=num_rays, finite_min_len=6, infinite_min_len=6, n_centres=3)
    
    # # print(f"ic: {ic.shape}, fc: {fc.shape}")
    # lidar_array = np.concatenate([np.array(ic)[:3], np.array(fc)[:3]], axis=1).flatten()
    # # print(f"lidar array: {lidar_array.shape}")
    # # Convert back to torch tensor (and copy sensor device if needed)
    # lidar_tensor = torch.tensor(lidar_array, dtype=torch.float32)
    # if torch.is_tensor(sensor_pos):
    #     lidar_tensor = lidar_tensor.to(sensor_pos.device)
    return lidar_tensor, mask
    # else:
    #     # --- Batched version: sensor_pos has shape (batch,2) ---
    #     batch_size = sensor_pos.shape[0]
    #     lidar_list = []
    #     for i in range(batch_size):
    #         # Process each sample individually.
    #         single_lidar, safe_mask = get_lidar_centres_tensor(sensor_pos[i],
    #                                                 bot_orientation[i] if torch.is_tensor(bot_orientation) else bot_orientation[i],
    #                                                 binary_img, world_limits, threshold, num_rays, max_range)
    #         lidar_list.append(single_lidar)
    #     return torch.stack(lidar_list, dim=0), safe_mask


######################################################################
# Simulate trajectory with collision avoidance loss (for training)
######################################################################
def simulate_trajectory_with_collision(net, initial_pos, goal_pos, binary_img, world_limits, max_steps=38, threshold=3.0):
    """
    Given batched (or single) starting and goal positions, simulate the trajectory.
    Converts inputs (if necessary) to torch tensors and ensures a 2D shape is used
    (i.e. shape (batch,2) for positions). The LiDAR centres (size=6) are appended to the
    six navigation features to form a 12-D input. Additionally, a collision penalty is computed
    based on the difference between the predicted heading (theta+delta_theta)
    and one of the safe candidate headings derived from the LiDAR centres.
    """
    device = "cuda"
    # --- Convert inputs to torch tensors (if needed) and ensure batched dimensions ---
    if not torch.is_tensor(initial_pos):
        initial_pos = torch.tensor(initial_pos, dtype=torch.float32).to(device)
    if not torch.is_tensor(goal_pos):
        goal_pos = torch.tensor(goal_pos, dtype=torch.float32).to(device)
    # For a single sample, add batch dimension.
    if initial_pos.ndim == 1:
        initial_pos = initial_pos.unsqueeze(0)
    if goal_pos.ndim == 1:
        goal_pos = goal_pos.unsqueeze(0)
    
    current_pos = initial_pos.clone()   # (batch, 2)
    batch_size = current_pos.shape[0]
    theta = torch.zeros(batch_size).to(device)  # initial heading (batch,)
    total_loss = torch.tensor(0.0).to(device)
    path_length = torch.zeros(batch_size).to(device)
    prev_controls = None
    eps = 1e-6

    # temporal_weight = 1.0  # initial discount weight

    for step in range(max_steps):
        delta_pos = goal_pos - current_pos  # (batch,2)
        # temporal_weight = 0.85 ** (step*-1)
        temporal_weight = 0.5 * torch.sqrt(torch.tensor(step))

        # Compute the angle to the goal for each sample.
        rel_angle = torch.atan2(delta_pos[:, 1], delta_pos[:, 0]).to(device)  # (batch,)
        rel_angle = (rel_angle + 2 * torch.pi) % (2 * torch.pi) 
        relative_theta = rel_angle - theta
        # Wrap angle to [-pi, pi]
        relative_theta = (relative_theta + 2 * torch.pi) % (2 * torch.pi)

        # --- Extract LiDAR centres ---
        lidar_centres, safe_mask = get_lidar_centres_tensor(current_pos, relative_theta, binary_img, world_limits, threshold)
        # lidar_centres = (lidar_centres + np.pi) % (2 * np.pi) - np.pi # [-pi, pi]
        safe_mask = torch.tensor(safe_mask).to(device)

        # Ensure lidar_features has the proper batch shape
        # if lidar_centres.ndim == 1:
            # lidar_features = (lidar_centres / torch.pi).unsqueeze(0)
        # else:
            # lidar_features = lidar_centres / torch.pi  # (batch,6)

        # --- Build Navigation Features ---
        net_input = torch.stack([
            current_pos[:, 0] / 10,
            current_pos[:, 1] / 7,
            goal_pos[:, 0]    / 10,
            goal_pos[:, 1]    / 7,
            theta           / (2*torch.pi),
            relative_theta  / (2*torch.pi)
        ], dim=1).to(device)  # (batch,6)

        # Concatenate the navigation features with the LiDAR centres features (12-D input)
        # net_input = torch.cat([nav_features, lidar_features], dim=1).to(device)  # (batch,12)

        # --- Network Prediction ---
        controls = net(net_input, safe_mask)[0]  # (batch,2); controls[:,0] = L, controls[:,1] = delta_theta (scaled)
        L = controls[:, 0]      # linear velocity (batch,)
        delta_theta = controls[:, 1]  # already in [0, 2pi] (batch,)
        # delta_theta_norm = delta_theta.clone().detach() + torch.pi # [0, 2pi]

        # --- Collision-Avoidance Penalty (Fixed for Batched Differences) ---
        # Compute desired heading for each sample as a single value, then unsqueeze to shape (batch, 1)
        # desired_heading = (theta + delta_theta).unsqueeze(1)  # shape: (batch, 1) 

        # Ensure lidar_centres is a torch tensor on the proper device.
        # if not torch.is_tensor(lidar_centres):
            # lidar_centres = torch.tensor(lidar_centres, dtype=torch.float32, device=desired_heading.device) # [batch, 6]
        # else:
            # lidar_centres = lidar_centres.to(desired_heading.device) # [batch, 6]

        # Compute the angular differences element-wise.
        # This gives a tensor of shape (batch, 6) with the differences between the desired heading and each LiDAR centre.
        # abs_diff = torch.abs(desired_heading - lidar_centres)
        # diff_stack = torch.minimum(abs_diff, 2 * torch.pi - abs_diff)

        # Apply a softmin over the 3 candidates so that lower differences are weighted higher.
        # collision_soft_weights = torch.nn.functional.softmin(diff_stack, dim=1)  # (batch, 6)

        # Compute a weighted collision penalty.
        # collision_penalty = (collision_soft_weights * diff_stack).sum(dim=1)  # (batch,)
        # collision_weight = 1.0
        # collision_loss = collision_weight * collision_penalty

        # --- Update heading and position ---
        theta = delta_theta
        theta = (theta + 2 * torch.pi) % (2 * torch.pi) 

        delta_x = L * torch.cos(theta)
        delta_y = L * torch.sin(theta)
        movement = torch.stack([delta_x, delta_y], dim=1)  # (batch,2)
        path_length = path_length + torch.norm(movement, dim=1)
        current_pos = current_pos + movement
        delta_pos = goal_pos - current_pos  # (batch,2)

        # Per-step losses: distance from goal and control magnitude
        step_position_loss = temporal_weight * torch.norm(delta_pos, dim=1)
        # control_loss = 0.1 * torch.abs(delta_theta)
        # print(f"Step_position_Loss: {step_position_loss.mean()}")
        step_loss = step_position_loss

        # print(f"Step Loss: {step_loss.mean()}")
        total_loss += step_loss.mean()
        # print(f"Total Loss: {total_loss}")

        # temporal_weight *= 0.85

    # Final losses: distance from goal and path efficiency
    final_pos_loss = 10.0 * torch.norm(current_pos - goal_pos, dim=1)
    straight_line_dist = torch.norm(goal_pos - initial_pos, dim=1)
    efficiency_loss = 0.8 * (path_length / (straight_line_dist + eps))
    total_loss += (final_pos_loss.mean() + efficiency_loss.mean())
    return total_loss


#############################################
# Batch generator (unchanged from your code)
#############################################

def gen_bot_positions(batch_size):
    # Precompute candidate x values
    all_x = np.linspace(-3, 5, 10000)
    exclude_x = np.concatenate([
        np.linspace(-2.8, -1.7, 900),
        np.linspace(-1.2, 0.5, 1200),
        np.linspace(1.5, 2.6, 900),
        np.linspace(3.2, 4.6, 1200)
    ])
    candidate_x = np.array(sorted(set(all_x) - set(exclude_x)))
    
    # Precompute candidate y values
    all_y = np.linspace(-7, 4, 14000)
    # exclude_y = np.concatenate([
    #     np.linspace(-1.5, 2.5, 1000),
    #     np.linspace(-2.5, -5.6, 3100)
    # ])
    # candidate_y = np.array(sorted(set(all_y) - set(exclude_y)))
    candidate_y = all_y
    
    # Sample batch_size positions for x and y independently
    x_samples = np.random.choice(candidate_x, size=batch_size, replace=True)
    y_samples = np.random.choice(candidate_y, size=batch_size, replace=True)
    
    # Combine the samples into an array of shape (batch_size, 2)
    positions = np.stack((x_samples, y_samples), axis=1)
    return positions

def generate_batch(batch_size, min_distance=0.5, max_distance=22):
    # Define channel dimensions for batch and vector components.
    batch_dim = batch(samples=batch_size)
    vec_dim = channel(vector='x,y')
    
    positions = gen_bot_positions(batch_size)
    
    initial_pos = math.tensor(positions, batch('samples'), channel(vector='x,y'))
    
    # Generate random displacement directions (angles)
    angle = math.random_uniform(batch_dim, low=-math.pi, high=math.pi)
    
    # Generate displacement magnitudes between [min_distance, max_distance]
    distance = math.random_uniform(batch_dim, low=min_distance, high=max_distance)
    
    # Compute displacement components using phi Flow math functions.
    dx = distance * math.cos(angle)
    dy = distance * math.sin(angle)
    
    # Apply displacement to initial positions.
    goal_x = initial_pos['x'] + dx
    goal_y = initial_pos['y'] + dy
    
    # Clamp goal positions to stay within defined bounds.
    goal_x = math.clip(goal_x, -7.9, 7.9)
    goal_y = math.clip(goal_y, -5.9, 5.9)
    
    # Stack the goal coordinates to create the goal position tensor.
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

def plot_trajectory_with_collision(net, initial_pos, goal_pos, binary_img, world_limits, epoch_no, max_steps=58, threshold=3.0):
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
        old_positions = [current_pos.cpu().numpy()]
        
        for stp in range(max_steps):
            delta_pos = goal_pos - current_pos
            rel_angle = torch.atan2(delta_pos[1:2], delta_pos[0:1])
            rel_angle = (rel_angle + 2 * torch.pi) % (2 * torch.pi)
            relative_theta = rel_angle - theta
            relative_theta = (relative_theta + 2 * torch.pi) % (2 * torch.pi) 

            # Get LiDAR centres from environment. This function handles both single and batched inputs.
            lidar_centres, safe_mask = get_lidar_centres_tensor(current_pos, theta, binary_img, world_limits, threshold=threshold)
            # lidar_features = lidar_centres / torch.pi
            safe_mask = torch.tensor(safe_mask).to(device)

            net_input = torch.stack([
                current_pos[0] / 10,
                current_pos[1] / 7,
                goal_pos[0] / 10,
                goal_pos[1] / 7,
                theta[0] / (2 * torch.pi),
                relative_theta[0] / (2 * torch.pi)
            ], dim=-1).to('cuda')
            # net_input = torch.cat([nav_features, lidar_features]).unsqueeze(0).to('cuda')
            
            controls = net(net_input, safe_mask)
            L = controls[0][0].detach().cpu()
            delta_theta = controls[0][1].detach().cpu()
            delta_theta_old = controls[1][1].detach().cpu()
            diff_theta = delta_theta - delta_theta_old

            # print(f"Theta_old: {np.rad2deg(delta_theta_old)}, Theta_New: {np.rad2deg(delta_theta)}, Diff_theta: {np.rad2deg(diff_theta)}")

            old_theta = delta_theta_old.unsqueeze(-1)
            theta = delta_theta.unsqueeze(-1)
            old_theta = (old_theta + 2 * torch.pi) % (2 * torch.pi) 
            theta = (theta + 2 * torch.pi) % (2 * torch.pi) 

            movement = torch.stack([L * torch.cos(theta), L * torch.sin(theta)]).squeeze()
            movement_old = torch.stack([L * torch.cos(old_theta), L * torch.sin(old_theta)]).squeeze()

            current_pos = current_pos + movement
            c_prime = current_pos + movement_old
            positions.append(current_pos.cpu().numpy())
            old_positions.append(c_prime.cpu().numpy())
            
            if torch.norm(delta_pos) < 0.1:
                break
        
        img = r"F:\Aerosim-Simulation-Zone\Try\New_WR_World.png"
        positions = np.array(positions)
        old_positions = np.array(old_positions)
        # print(f"Old_Positions : {old_positions}")
        plt.figure(figsize=(8, 6))
        plt.imshow(binary_img, extent=[-10, 10, -8, 8], cmap='gray')
        plt.plot(positions[:, 0], positions[:, 1], 'b-o', markersize=4, label='Path')
        # plt.scatter(old_positions[:, 0], old_positions[:, 1], 'b-o', markersize=4, label='old_Path')
        plt.scatter(positions[0, 0], positions[0, 1], c='green', s=200, marker='*', label='Start')
        plt.scatter(goal_pos[0].item(), goal_pos[1].item(), c='red', s=200, marker='X', label='Goal')
        plt.title("Navigation Trajectory with Collision Avoidance")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show(block=False)
        # plt.imshow(binary_img)
        plt.savefig(os.path.join(r"F:\Aerosim-Simulation-Zone\Try\FIGS_Bound", f"Next_Train_{epoch_no}.png"))
        plt.close('all')


#############################################
# TRAINING LOOP
#############################################
device="cuda"
model = r"F:\Aerosim-Simulation-Zone\Try\FIGS_Bound\checkpoint_epoch_22000.pth"
net = torch.load(model, weights_only=False).to(device)
# net = NavigationNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# Load the binary image used in LiDAR collision simulation.
binary_img = create_binary_image(image_path=r"F:\Aerosim-Simulation-Zone\Try\World_Map.png")
# Define world boundaries (adjust as needed)
world_limits = np.array([[-10, 10], [-8, 8]])
batch_size = 512
epochs = 20001
base_epoch = 22000
pbar = tqdm.tqdm(range(base_epoch, base_epoch+epochs), desc="Training", dynamic_ncols=True) 

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
    
    if epoch % 100 == 0:
        with torch.no_grad():
            # Generate new single-sample test positions.
            initial_pos, goal_pos = generate_batch(1, 4)
            initial_pos = initial_pos.native("samples,vector").squeeze(0)
            goal_pos = goal_pos.native("samples,vector").squeeze(0)
            test_loss = physics_collision_loss(net, initial_pos, goal_pos, binary_img, world_limits)
            print(f"Epoch {epoch}, Loss: {test_loss.item():.4f}")
            plot_trajectory_with_collision(net, initial_pos, goal_pos, binary_img, world_limits, epoch_no=epoch, max_steps=20, threshold=3.0)
    if epoch % 500 == 0:
        torch.save(net, f'F:\Aerosim-Simulation-Zone\Try\FIGS_Bound\checkpoint_epoch_{epoch}.pth')

torch.save(net, '500_coll_nav_model.pth')

# # Finally, test on one sample and save the model.
# with torch.no_grad():
#     initial_torch = torch.tensor([4.8, 6], dtype=torch.float32)
#     goal_torch = torch.tensor([-3.1, 0.5], dtype=torch.float32)
#     test_loss = physics_collision_loss(net, initial_torch, goal_torch, binary_img, world_limits)
#     plot_trajectory_with_collision(net, initial_torch, goal_torch, binary_img, world_limits, epoch_no=epoch, max_steps=20, threshold=3.0)
