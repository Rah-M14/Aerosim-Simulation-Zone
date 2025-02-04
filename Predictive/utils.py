import torch
import numpy as np
import matplotlib.pyplot as plt
from phi.torch.flow import *
from typing import Tuple, Optional

class NavigationUtils:
    def __init__(self, world_bounds: Tuple[float, float] = (8.0, 6.0)):
        self.world_bounds = world_bounds

    @staticmethod
    def generate_batch(batch_size: int, min_distance: float = 2.0, max_distance: float = 12.0):
        """Generate a batch of initial and goal positions"""
        batch_dim = batch(samples=batch_size)
        vec_dim = channel(vector='x,y')
        
        # Generate initial positions
        initial_x = math.random_uniform(batch_dim, low=-8, high=8)
        initial_y = math.random_uniform(batch_dim, low=-6, high=6)
        initial_pos = math.stack([initial_x, initial_y], vec_dim)
        
        # Generate random displacement direction
        angle = math.random_uniform(batch_dim, low=-math.pi, high=math.pi)
        
        # Generate displacement magnitudes
        distance = math.random_uniform(batch_dim, low=min_distance, high=max_distance)
        
        # Compute displacement components
        dx = distance * math.cos(angle)
        dy = distance * math.sin(angle)
        
        # Apply displacement and clamp to bounds
        goal_x = math.clip(initial_x + dx, -7.9, 7.9)
        goal_y = math.clip(initial_y + dy, -5.9, 5.9)
        
        goal_pos = math.stack([goal_x, goal_y], vec_dim)
        
        return initial_pos, goal_pos

    @staticmethod
    def plot_trajectory(net, initial_pos, goal_pos, max_steps: int = 12, 
                       save_path: Optional[str] = None, show: bool = True):
        """Simulate and plot a single trajectory"""
        with torch.no_grad():
            current_pos = initial_pos.clone()
            goal_pos = goal_pos.clone()
            theta = torch.zeros_like(current_pos[0])
            positions = []
            
            for step in range(max_steps):
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
                
                theta = (theta + delta_theta + np.pi) % (2 * np.pi) - np.pi
                
                movement = torch.stack([
                    L * torch.cos(theta),
                    L * torch.sin(theta)
                ])
                
                current_pos += movement
                positions.append(current_pos.cpu().numpy().copy())
                
                if torch.norm(delta_pos) < 0.1:
                    break
            
            # Create visualization
            positions = np.array(positions)
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot trajectory and points
            ax.plot(positions[:, 0], positions[:, 1], 'b-o', markersize=4, label='Path')
            ax.scatter(positions[0, 0], positions[0, 1], c='green', s=200, marker='*', label='Start')
            ax.scatter(goal_pos[0].item(), goal_pos[1].item(), c='red', s=200, marker='X', label='Goal')
            
            # Customize plot
            ax.set_title("Navigation Trajectory")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.grid(True)
            ax.set_aspect('equal', adjustable='box')
            
            # Set bounds
            max_bound = max(8.0, 6.0)  # Using default world bounds
            ax.set_xlim([-max_bound, max_bound])
            ax.set_ylim([-max_bound, max_bound])
            
            ax.legend(loc='upper right')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            if show:
                plt.show()
            else:
                plt.close()
            
            return fig

    @staticmethod
    def save_model(net, path: str):
        """Save model to disk"""
        torch.save(net.state_dict(), path)

    @staticmethod
    def load_model(net, path: str):
        """Load model from disk"""
        net.load_state_dict(torch.load(path))
        return net