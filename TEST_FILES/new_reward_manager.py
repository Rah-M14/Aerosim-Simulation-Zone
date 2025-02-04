import numpy as np
from typing import Dict, Any

class RewardManager:
    def __init__(self, world_theta: float, monitor: Any = None):

        # World Setup
        self.world_theta = world_theta
        self.world_limits = np.array([[-8, 8], [-6, 6]])
        self.monitor = monitor


        # Core Rews
        self.GOAL_REACHED_REWARD = 5000.0
        self.BOUNDARY_PENALTY = -1000.0
        self.TIMEOUT_PENALTY = -1000.0
        self.LIVE_REWARD = -1.0
        
        # Potential Components
        self.GAMMA = 0.99
        self.ALPHA_GOAL = 0.7
        self.ALPHA_PATH = 0.3
        
        # Progress Rews
        self.PROGRESS_BASE = 10.0
        self.PATH_FOLLOW_BASE = 1.0
        self.HEADING_BASE = 10.0
        
        # Penalties
        self.DEVIATION_PENALTY_BASE = -10.0
        self.OSCILLATION_PENALTY_BASE = -10.0
        
        # Thresholds
        self.MIN_PROGRESS_THRESHOLD = 0.01
        self.MAX_DEVIATION_THRESHOLD = 1.0
        self.MAX_HISTORY = 5
        
        # History Tracking
        self.prev_goal_potential = None
        self.prev_path_potential = None
        self.position_history = []
        self.reward_history = []

    def reset(self):
        self.prev_goal_potential = None
        self.prev_path_potential = None
        self.position_history = []
        self.reward_history = []
        
    def compute_reward(self, current_pos: np.ndarray, prev_pos: np.ndarray, 
                      goal_pos: np.ndarray, chunk: np.ndarray, world_theta: float, timestep: int) -> float:
        reward = 0.0
        
        goal_pot_rew, goal_pot = self.goal_pot_rew(current_pos, goal_pos, timestep)
        path_pot_rew, path_pot = self.path_pot_rew(current_pos, chunk)
        
        progress_rew, progress = self.progress_rew(prev_pos, current_pos, chunk, goal_pos, timestep)
        path_follow_rew = self.path_follow_rew(current_pos, progress, chunk, timestep)
        heading_rew = self.heading_rew(current_pos, goal_pos, world_theta, timestep)
                
        oscillation_penalty = self.oscillation_penalty(current_pos)

        reward = goal_pot_rew + path_pot_rew + progress_rew + path_follow_rew + heading_rew + oscillation_penalty + self.LIVE_REWARD 
        
        # History Update
        self.prev_goal_potential = goal_pot
        self.prev_path_potential = path_pot
        triple = np.concatenate([current_pos, np.array([world_theta])])
        self.position_history.append(triple)
        if len(self.position_history) > self.MAX_HISTORY:
            self.position_history.pop(0)
        self.reward_history.append(reward)
        
        reward_components = {
            'total_reward': reward,
            'goal_potential': goal_pot_rew,
            'path_potential': path_pot_rew,
            'progress': progress_rew,
            'path_following': path_follow_rew,
            'heading': heading_rew,
            'oscillation_penalty': oscillation_penalty
        }
        
        # Update the monitor if it exists
        if hasattr(self, 'monitor'):
            self.monitor.update(reward_components)
        
        return reward, reward_components
    
    # def goal_pot_rew(self, current_pos: np.ndarray, goal_pos: np.ndarray) -> float:
    #     current_goal_potential = self._goal_potential(current_pos, goal_pos)        
    #     if self.prev_goal_potential is None:
    #         self.prev_goal_potential = current_goal_potential

    #     goal_shaping = self.GAMMA * self.prev_goal_potential - current_goal_potential
    #     return self.ALPHA_GOAL * goal_shaping, -current_goal_potential

    def goal_pot_rew(self, current_pos: np.ndarray, goal_pos: np.ndarray, timestep: int) -> float:
        current_goal_potential = self._goal_potential(current_pos, goal_pos)
        if self.prev_goal_potential is None:
            self.prev_goal_potential = current_goal_potential
        diff = self.prev_goal_potential - current_goal_potential

        if (timestep > 100 and diff <= 0):
            return -current_goal_potential*(timestep/100), current_goal_potential
        else:
            return -current_goal_potential + diff, current_goal_potential
    
    def path_pot_rew(self, current_pos: np.ndarray, chunk: np.ndarray) -> float:
        current_path_potential = self._path_potential(current_pos, chunk)
        if self.prev_path_potential is None:
            self.prev_path_potential = current_path_potential

        path_shaping = self.GAMMA * self.prev_path_potential - current_path_potential
        return self.ALPHA_PATH * path_shaping, -current_path_potential
    
    def progress_rew(self, prev_pos: np.ndarray, current_pos: np.ndarray, chunk: np.ndarray, goal_pos: np.ndarray, timestep: int) -> float:
        progress = self._compute_progress(prev_pos, current_pos, chunk, goal_pos)
        if progress > self.MIN_PROGRESS_THRESHOLD:
            return self.PROGRESS_BASE * progress, progress
        return -5 * (timestep/100), 0.0
    
    def path_follow_rew(self, current_pos: np.ndarray, progress: float, chunk: np.ndarray, timestep: int) -> float:
        path_deviation = self._compute_path_deviation(current_pos, chunk)
        if path_deviation > self.MAX_DEVIATION_THRESHOLD:
            adaptive_scale = abs(self.MAX_DEVIATION_THRESHOLD - path_deviation)
            path_follow_reward = self.DEVIATION_PENALTY_BASE * path_deviation * adaptive_scale
            return path_follow_reward
        elif progress > self.MIN_PROGRESS_THRESHOLD:
            path_follow_reward = self.PATH_FOLLOW_BASE * np.exp((4/self.MAX_DEVIATION_THRESHOLD) * path_deviation)
            return path_follow_reward * (timestep/100)
        else:
            return 0.0
        
    def heading_rew(self, current_pos: np.ndarray, goal_pos: np.ndarray, world_theta: float, timestep: int) -> float:
        desired_heading = self._compute_desired_heading(current_pos, goal_pos)
        heading_alignment = self._compute_heading_alignment(world_theta, desired_heading)
        return self.HEADING_BASE * heading_alignment * (timestep/100) if heading_alignment < 0.0 else 0.0
    
    def oscillation_penalty(self, current_pos: np.ndarray) -> float:
        if len(self.position_history) >= 3:
            oscillation = self._compute_oscillation(current_pos)
            detection = self._detect_oscillation(self.position_history)
            penalty = 0.0
            if detection['is_oscillating']:
                penalty = detection['heading_variance'] + (detection['cumulative_curvature'] - 5)
            return (self.OSCILLATION_PENALTY_BASE * oscillation) - np.clip(penalty, 0.0, 100.0)
        return 0.0
    

    
    def _goal_potential(self, current_pos: np.ndarray, goal_pos: np.ndarray) -> float:
        dist = np.linalg.norm(current_pos - goal_pos)
        return dist 
    
    def _path_potential(self, current_pos: np.ndarray, chunk: np.ndarray) -> float:
        # distances = [np.linalg.norm(current_pos - waypoint) for waypoint in chunk]
        dist_line = self._point_to_line_segment_distance(current_pos, chunk[0], chunk[1])
        dist_point_1 = np.linalg.norm(current_pos - chunk[0])
        dist_point_2 = np.linalg.norm(current_pos - chunk[1])
        path_potential = dist_line + min(dist_point_1, dist_point_2)
        return path_potential
    
    def _compute_progress(self, prev_pos: np.ndarray, current_pos: np.ndarray, 
                         target_pos: np.ndarray, goal_pos: np.ndarray) -> float:
        prev_dist = {"0": np.linalg.norm(prev_pos - target_pos[0]),
                     "1": np.linalg.norm(prev_pos - target_pos[1])}
        current_dist = {"0": np.linalg.norm(current_pos - target_pos[0]),
                        "1": np.linalg.norm(current_pos - target_pos[1])}
        goal_cur_dist = np.linalg.norm(goal_pos - current_pos)
        goal_prev_dist = np.linalg.norm(goal_pos - prev_pos)
        diff_g = goal_prev_dist - goal_cur_dist
        
        prev_dist = min(prev_dist["0"], prev_dist["1"])
        current_dist = min(current_dist["0"], current_dist["1"])
        diff_c = prev_dist - current_dist

        progress = 0.35 * diff_c + 0.65 * diff_g
        return progress
    
    def _compute_path_deviation(self, current_pos: np.ndarray, chunk: np.ndarray) -> float:
        if len(chunk) < 2:
            return np.linalg.norm(current_pos - chunk[0])
            
        min_dist = float('inf')
        for i in range(len(chunk) - 1):
            dist = self._point_to_line_segment_distance(current_pos, chunk[i], chunk[i+1])
            min_dist = min(min_dist, dist)
        return min_dist
    
    def _compute_desired_heading(self, current_pos: np.ndarray, target_pos: np.ndarray) -> float:
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        return np.arctan2(dy, dx)
    
    def _compute_heading_alignment(self, world_theta: float, desired_heading: float) -> float:
        current_heading = world_theta
        angle_diff = abs(current_heading - desired_heading)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
        return np.cos(angle_diff)
    
    def _compute_oscillation(self, current_pos: np.ndarray) -> float:
        if len(self.position_history) < 3:
            return 0.0
            
        vectors = []
        for i in range(len(self.position_history)-1):
            v = self.position_history[i+1] - self.position_history[i]
            vectors.append(v / (np.linalg.norm(v) + 1e-6))
            
        dot_products = [np.dot(vectors[i], vectors[i+1]) 
                       for i in range(len(vectors)-1)]
        return abs(np.mean(dot_products))
    
    def _detect_oscillation(self, history):
        states = np.array(history)

        # Calculate distances between consecutive states
        distances = np.linalg.norm(np.diff(states[:,0:2], axis=0))
        distance_variance = np.var(distances)

        # Calculate orientation changes (wrapped to [-pi, pi])
        delta_theta = np.diff(states[:, 2])
        delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
        heading_variance = np.var(delta_theta)

        # Calculate curvature
        curvatures = np.abs(delta_theta) / (distances + 1e-6)
        cumulative_curvature = np.sum(curvatures)

        # Oscillation detection heuristic: high variance in distance or heading, or high curvature
        is_oscillating = (
            distance_variance > 0.05 or  # Tunable threshold for distance variance
            heading_variance > 0.1 or   # Tunable threshold for heading variance
            cumulative_curvature > 5    # Tunable threshold for curvature
        )
        return {
            'distance_variance': distance_variance,
            'heading_variance': heading_variance,
            'cumulative_curvature': cumulative_curvature,
            'is_oscillating': is_oscillating
        }
    
    def _point_to_line_segment_distance(self, point: np.ndarray, 
                                      line_start: np.ndarray, 
                                      line_end: np.ndarray) -> float:
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_length = np.linalg.norm(line_vec)
        point_vec_length = np.linalg.norm(point_vec)
        
        if line_length == 0:
            return point_vec_length
            
        t = max(0, min(1, np.dot(point_vec, line_vec) / (point_vec_length * line_length)))
        projection = line_start + t * line_vec
        return np.linalg.norm(point - projection)
    
    def out_of_boundary_penalty(self, current_pos: np.ndarray) -> float:
        x, y = current_pos
        if x < self.world_limits[0][0] or x > self.world_limits[0][1] or y < self.world_limits[1][0] or y > self.world_limits[1][1]:
            return self.BOUNDARY_PENALTY
        return 0.0