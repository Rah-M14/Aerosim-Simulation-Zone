# %%
import torch
from phi.torch.flow import *

class NavigationNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.control_net = torch.nn.Sequential(
            torch.nn.Linear(6, 16),  # Input: [bot_x, bot_y, goal_x, goal_y, world_theta, relative_theta]
            torch.nn.Tanh(),
            torch.nn.Linear(16, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 2), # Output: [L, delta_theta]
            torch.nn.Tanh()  # Tanh activation for bounded output
        )
        
    def forward(self, current_state):
        return self.control_net(current_state)

# %%
def simulate_trajectory(net, initial_pos, goal_pos, max_steps=12):

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    current_pos = initial_pos
    theta = math.zeros(initial_pos.shape.non_channel)
    total_loss = 0
    path_length = 0  # Track total distance traveled
    prev_controls = None  # For control smoothness
    
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
            current_pos.vector['x']/10, 
            current_pos.vector['y']/7,
            goal_pos.vector['x']/10,
            goal_pos.vector['y']/7,
            theta/math.PI,
            relative_theta/math.PI
        ], channel('input_features'))
        
        # Network prediction
        controls = math.native_call(net.to(device), net_input)
        L = controls.vector[0]
        delta_theta = controls.vector[1]*math.PI

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

    t_loss = total_loss + math.mean(final_pos_loss + efficiency_loss)
    print(f"Total Loss: {t_loss}")
    
    return t_loss


# %%
# from phi.torch.flow import batch, channel

def generate_batch(batch_size, min_distance=2.0, max_distance=12):
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

# %%
def physics_loss(net, initial_pos, goal_pos):
    return simulate_trajectory(net, initial_pos, goal_pos)

# %%
import matplotlib.pyplot as plt
import torch

def plot_trajectory(net, initial_pos, goal_pos, max_steps=12):
    """Simulate and plot a single trajectory using PyTorch tensors"""
    with torch.no_grad():
        current_pos = initial_pos.clone()
        goal_pos = goal_pos.clone()

        print(current_pos, goal_pos)
        
        theta = torch.zeros_like(current_pos[0])
        positions = []
        
        for stp in range(max_steps):
            delta_pos = goal_pos - current_pos
            relative_theta = torch.atan2(delta_pos[1], delta_pos[0]) - theta

            relative_theta = (relative_theta + np.pi) % (2 * np.pi) - np.pi
            
            net_input = torch.stack([
                current_pos[0]/8, current_pos[1]/6,
                goal_pos[0]/8, goal_pos[1]/6,
                theta/math.PI, relative_theta/math.PI
            ], dim=-1).unsqueeze(0)
            
            controls = net(net_input)[0]
            L = controls[0]
            delta_theta = controls[1]*math.PI

            # print(L, delta_theta, theta, torch.clip(delta_theta, -math.PI/9, math.PI/9))

            theta = theta + delta_theta
            theta = (theta + np.pi) % (2 * np.pi) - np.pi 


            # print("step", stp, " ", np.rad2deg(theta), net_input)
        
            
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
        plt.scatter(positions[0, 0], positions[0, 1], c='green', s=200, marker='*', label='Start')
        plt.scatter(goal_pos[0].item(), goal_pos[1].item(), c='red', s=200, marker='X', label='Goal')
        plt.title("Navigation Trajectory")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()


# %%
import numpy as np

net = NavigationNet()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(10000):
    # Generate batch using PhiFlow
    initial_pos, goal_pos = generate_batch(512, min_distance=2.0, max_distance=10.0)

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

            plot_trajectory(net, initial_torch, goal_torch)

# %%
with torch.no_grad():
    initial_torch = torch.tensor([4.5, -3.2], dtype=torch.float32)
    goal_torch = torch.tensor([2.1, -6.1], dtype=torch.float32)

    loss = physics_loss(net, initial_torch, goal_torch)
    # print(f"Epoch {epoch}, Loss: {loss.native().item():.4f}")

    plot_trajectory(net, initial_torch, goal_torch, 60)

# %%
torch.save(net, 'nav_model.pth')


