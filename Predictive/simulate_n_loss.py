from phi.torch.flow import *
import torch

class TrajectorySimulator:
    def __init__(self, 
                 temporal_discount: float = 0.85,
                 control_change_weight: float = 0.25,
                 angular_control_weight: float = 0.1,
                 final_position_weight: float = 10.0,
                 efficiency_weight: float = 0.9):
        self.temporal_discount = temporal_discount
        self.control_change_weight = control_change_weight
        self.angular_control_weight = angular_control_weight
        self.final_position_weight = final_position_weight
        self.efficiency_weight = efficiency_weight
        self.eps = 1e-6

    def simulate_trajectory(self, net, initial_pos, goal_pos, max_steps=12):
        device = "cuda" if torch.cuda.is_available() else 'cpu'

        current_pos = initial_pos
        theta = math.zeros(initial_pos.shape.non_channel)
        total_loss = 0
        path_length = 0
        prev_controls = None
        trajectory = [current_pos]
        
        for step in range(max_steps):
            delta_pos = goal_pos - current_pos
            temporal_weight = self.temporal_discount ** (step*-1)
            
            # Calculate relative angle
            relative_theta = math.arctan(delta_pos.vector['y'], 
                                       divide_by=delta_pos.vector['x']+self.eps) - theta
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
            
            # Get controls from network
            controls = math.native_call(net.to(device), net_input)
            L = controls.vector[0]
            delta_theta = controls.vector[1]*math.PI

            # Control smoothness loss
            if prev_controls is not None:
                control_change = math.vec_squared(controls - prev_controls)
                total_loss += self.control_change_weight * math.mean(control_change)
            prev_controls = controls

            # Update orientation
            theta += delta_theta
            theta = (theta + np.pi) % (2 * np.pi) - np.pi 
            
            # Calculate movement
            movement = math.stack([
                L * math.cos(theta),
                L * math.sin(theta)
            ], dim=channel(vector='x,y'))

            # Update path length and position
            path_length += math.vec_length(movement)
            new_pos = current_pos + movement
            trajectory.append(new_pos)

            # Calculate losses
            position_loss = temporal_weight * math.vec_length(delta_pos)
            control_loss = self.angular_control_weight * math.abs(delta_theta)
            total_loss += math.mean(position_loss + control_loss)
            
            # Update position if not close enough to goal
            current_pos = math.where(math.vec_length(delta_pos) > 0.1, new_pos, current_pos)
        
        # Final position loss
        final_pos_loss = self.final_position_weight * math.vec_length(trajectory[-1] - goal_pos)
        
        # Path efficiency loss
        straight_line_dist = math.vec_length(goal_pos - initial_pos)
        efficiency_loss = self.efficiency_weight * (path_length / (straight_line_dist + self.eps))

        return total_loss + math.mean(final_pos_loss + efficiency_loss)

    def physics_loss(self, net, initial_pos, goal_pos):
        """Wrapper for simulate_trajectory to be used as a loss function"""
        return self.simulate_trajectory(net, initial_pos, goal_pos)