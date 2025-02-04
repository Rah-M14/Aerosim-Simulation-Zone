import numpy as np

class TrajLoss:
    def __init__(self, network, max_steps=12, temporal_discount=0.85):
        self.network = network
        self.max_steps = max_steps
        self.temporal_discount = temporal_discount
        self.eps = 1e-6
        
        # Loss weights
        self.control_change_weight = 0.25
        self.angular_control_weight = 0.1
        self.final_position_weight = 10.0
        self.efficiency_weight = 0.9
        self.world_bounds = [8, 6]

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _get_network_input(self, current_pos, goal_pos, theta, relative_theta):
        """Prepare normalized network input"""
        return np.array([
            current_pos[0]/self.world_bounds[0],  # x
            current_pos[1]/self.world_bounds[1],   # y
            goal_pos[0]/self.world_bounds[0],     # goal x
            goal_pos[1]/self.world_bounds[1],      # goal y
            theta/np.pi,
            relative_theta/np.pi
        ])

    def simulate_trajectory(self, initial_pos, goal_pos):
        """Simulate robot trajectory and calculate loss"""
        current_pos = np.array(initial_pos)
        theta = 0.0
        total_loss = 0.0
        path_length = 0.0
        prev_controls = None
        trajectory = [current_pos.copy()]

        for step in range(self.max_steps):
            # Calculate position delta and relative angle
            delta_pos = goal_pos - current_pos
            temporal_weight = self.temporal_discount ** (-step)
            
            relative_theta = self._normalize_angle(
                np.arctan2(delta_pos[1], delta_pos[0] + self.eps) - theta
            )

            # Get network predictions
            net_input = self._get_network_input(current_pos, goal_pos, theta, relative_theta)
            controls = self.network(net_input)
            # Ensure controls is properly shaped
            if isinstance(controls, np.ndarray):
                # print(controls.shape)
                if len(controls.shape) == 2:
                    controls = controls[0]  # Take first row if 2D
            
            L, delta_theta = float(controls[0]), float(controls[1]) * np.pi

            # Control smoothness loss
            if prev_controls is not None:
                control_change = np.sum((controls - prev_controls) ** 2)
                total_loss += self.control_change_weight * control_change
            prev_controls = controls.copy()

            # Update orientation
            theta = self._normalize_angle(theta + delta_theta)
            
            # Calculate movement
            movement = np.array([
                L * np.cos(theta),
                L * np.sin(theta)
            ])

            # Update position and track path
            path_length += np.linalg.norm(movement)
            new_pos = current_pos + movement
            trajectory.append(new_pos.copy())

            # Position and control losses
            position_loss = temporal_weight * np.linalg.norm(delta_pos)
            control_loss = self.angular_control_weight * abs(delta_theta)
            total_loss += position_loss + control_loss

            # Update position if not close enough to goal
            if np.linalg.norm(delta_pos) > 0.1:
                current_pos = new_pos

        # Final position loss
        final_pos_loss = self.final_position_weight * np.linalg.norm(trajectory[-1] - goal_pos)
        
        # Path efficiency loss
        straight_line_dist = np.linalg.norm(goal_pos - initial_pos)
        efficiency_loss = self.efficiency_weight * (path_length / (straight_line_dist + self.eps))

        return total_loss + final_pos_loss + efficiency_loss
