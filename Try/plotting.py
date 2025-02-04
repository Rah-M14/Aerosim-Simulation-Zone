import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Tuple
try:
    import seaborn as sns
    sns.set_style("whitegrid")  # Set seaborn style
except ImportError:
    plt.style.use('default')    # Fallback to default style

class TrajectoryVisualizer:
    def __init__(self, world_bounds: Tuple[float, float] = (8.0, 6.0)):
        self.world_bounds = world_bounds
        
    def _normalize_angle(self, angle: torch.Tensor) -> torch.Tensor:
        """Normalize angle to [-π, π]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def plot_trajectory(self, net, initial_pos: torch.Tensor, goal_pos: torch.Tensor, max_steps: int = 15):
        """Simulate and plot a single trajectory using PyTorch tensors"""
        with torch.no_grad():
            current_pos = initial_pos.clone()
            goal_pos = goal_pos.clone()
            theta = torch.zeros_like(current_pos[0])
            positions = [current_pos.cpu().numpy().copy()]
            
            for step in range(max_steps):
                # Calculate relative position and angle
                delta_pos = goal_pos - current_pos
                relative_theta = torch.atan2(delta_pos[1], delta_pos[0]) - theta
                relative_theta = self._normalize_angle(relative_theta)
                
                # Prepare network input
                net_input = torch.stack([
                    current_pos[0]/self.world_bounds[0],
                    current_pos[1]/self.world_bounds[1],
                    goal_pos[0]/self.world_bounds[0],
                    goal_pos[1]/self.world_bounds[1],
                    theta/np.pi,
                    relative_theta/np.pi
                ], dim=-1).unsqueeze(0)
                
                # Get network predictions
                controls = net(net_input)[0]
                L = controls[0]
                delta_theta = controls[1] * np.pi
                
                # Update orientation
                theta = self._normalize_angle(theta + delta_theta)
                
                # Calculate and apply movement
                movement = torch.stack([
                    L * torch.cos(theta),
                    L * torch.sin(theta)
                ])
                current_pos += movement
                
                # Store position
                positions.append(current_pos.cpu().numpy().copy())
                
                # Check if goal reached
                if torch.norm(delta_pos) < 0.1:
                    break
            
            # Create plot
            positions = np.array(positions)
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot trajectory and points
            ax.plot(positions[:, 0], positions[:, 1], 'b-o', markersize=4, label='Path')
            ax.scatter(positions[0, 0], positions[0, 1], c='green', s=200, marker='*', label='Start')
            ax.scatter(goal_pos[0].item(), goal_pos[1].item(), c='red', s=200, marker='X', label='Goal')
            
            # Add plot details
            ax.set_title("Navigation Trajectory")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.grid(True)
            
            # Set equal aspect ratio and limits properly
            max_bound = max(self.world_bounds)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim([-max_bound, max_bound])
            ax.set_ylim([-max_bound, max_bound])
            
            # Add legend
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            return fig

    def plot_multiple_trajectories(self, net, initial_positions: torch.Tensor, 
                                 goal_positions: torch.Tensor, max_steps: int = 15):
        """Plot multiple trajectories on the same figure"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        with torch.no_grad():
            for init_pos, goal_pos in zip(initial_positions, goal_positions):
                current_pos = init_pos.clone()
                theta = torch.zeros_like(current_pos[0])
                positions = [current_pos.cpu().numpy().copy()]
                
                for _ in range(max_steps):
                    delta_pos = goal_pos - current_pos
                    relative_theta = self._normalize_angle(
                        torch.atan2(delta_pos[1], delta_pos[0]) - theta
                    )
                    
                    net_input = torch.stack([
                        current_pos[0]/self.world_bounds[0],
                        current_pos[1]/self.world_bounds[1],
                        goal_pos[0]/self.world_bounds[0],
                        goal_pos[1]/self.world_bounds[1],
                        theta/np.pi,
                        relative_theta/np.pi
                    ], dim=-1).unsqueeze(0)
                    
                    controls = net(net_input)[0]
                    L = controls[0]
                    delta_theta = controls[1] * np.pi
                    
                    theta = self._normalize_angle(theta + delta_theta)
                    movement = torch.stack([
                        L * torch.cos(theta),
                        L * torch.sin(theta)
                    ])
                    
                    current_pos += movement
                    positions.append(current_pos.cpu().numpy().copy())
                    
                    if torch.norm(delta_pos) < 0.1:
                        break
                
                positions = np.array(positions)
                ax.plot(positions[:, 0], positions[:, 1], '-o', markersize=2, alpha=0.5)
                ax.scatter(init_pos[0].item(), init_pos[1].item(), c='green', s=100, marker='*')
                ax.scatter(goal_pos[0].item(), goal_pos[1].item(), c='red', s=100, marker='X')
        
        # Set plot properties
        ax.set_title("Multiple Navigation Trajectories")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True)
        
        # Set equal aspect ratio and limits properly
        max_bound = max(self.world_bounds)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-max_bound, max_bound])
        ax.set_ylim([-max_bound, max_bound])
        
        plt.tight_layout()
        return fig
