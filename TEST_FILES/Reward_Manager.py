import numpy as np
from typing import Tuple, Union
from configs import RewardConfig
import wandb

rew_config = RewardConfig()

class RewardManager:
    def __init__(self):
        # Reward/Penalty constants
        self.SUCCESS_REWARD = 20000.0
        self.TIMEOUT_PENALTY = -1000.0
        self.LIVE_REWARD = -1.0
        self.VISITING_REWARD = 10.0
        self.THETA_OVERDRIVE_PENALTY = -10.0

        # PATH REWARDS
        self.PATH_FOLLOWING_FACTOR = 10
        self.PATH_CORRIDOR_WIDTH = 0.5
        self.MAX_PATH_DEVIATION = 1.0
        self.DEVIATION_PENALTY_FACTOR = -5

        # Weights for different reward components
        self.CHUNK_DEVIATION_FACTOR = -10.0
        self.GOAL_DISTANCE_FACTOR = -5.0
        self.DIRECTION_SCALE_FACTORS = rew_config.DIRECTION_SCALE_FACTORS

        self.PROGRESS_WEIGHT = 1.0
        self.PATH_FOLLOWING_WEIGHT = 2.0
        self.ACTION_SMOOTHNESS_WEIGHT = 0.1
        
        # Thresholds
        self.WAYPOINT_THRESHOLD = 0.1
        self.GOAL_THRESHOLD = 0.1
        self.MAX_PATH_DEVIATION = 0.5

        # BOUNDARY LIMITS
        self.BOUNDARY_LIMITS = np.array([[-8, 8], [-6, 6]])
        self.BOUNDARY_PENALTY = -10000.0

        # Action smoothness parameters
        self.L_SMOOTHNESS_PENALTY = -10.0
        self.THETA_SMOOTHNESS_PENALTY = -30.0
        self.ALLOWABLE_L_CHANGE = 0.15  
        self.ALLOWABLE_THETA_CHANGE = 0.15  
        self.MAX_PENALTY_THRESHOLD = 1.0 
        self.prev_L = None
        self.prev_theta = None

        self.reward = 0
        self.episode_reward = 0

    def ep_log(self, episode_num):
        # wandb.log({"Learning_Curve": self.episode_reward, "episode_num": episode_num})
        self.episode_reward = 0
        return None

    def compute_reward(
        self,
        current_pos: np.ndarray,
        prev_pos: np.ndarray,
        goal_pos: np.ndarray,
        chunk: np.ndarray,
        action: np.ndarray  # [L, theta]
    ) -> float:
        self.reward = 0
        simple_reward = self.simple_reward(current_pos, goal_pos, chunk)
        path_reward = self.path_following_reward(current_pos, chunk)
        cosine_reward = self.scale_direction_reward(current_pos, prev_pos, goal_pos)
        vis_reward = self.visiting_reward(current_pos, chunk)
        smoothness_reward = self.action_smoothness_reward(action)

        self.reward = simple_reward + path_reward + cosine_reward + vis_reward + smoothness_reward + self.LIVE_REWARD
        self.episode_reward += self.reward

        return self.reward
    
    def simple_reward(self, current_pos, goal_pos, chunk):
        n_dist = np.linalg.norm(chunk[0] - current_pos)
        return n_dist*self.CHUNK_DEVIATION_FACTOR if n_dist > self.WAYPOINT_THRESHOLD else 0.0
    
    def reward_for_direction(self, current_pos, prev_pos, goal_pos):
        vector_to_goal = np.array(goal_pos) - np.array(prev_pos)
        vector_action = np.array(current_pos) - np.array(prev_pos)
        cosine_similarity = np.dot(vector_to_goal, vector_action) / (np.linalg.norm(vector_to_goal) * np.linalg.norm(vector_action) + 1e-9)
        return cosine_similarity

    def scale_direction_reward(self, current_pos, prev_pos, goal_pos):
        cosine_similarity = self.reward_for_direction(current_pos, prev_pos, goal_pos)
        distance_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(current_pos))

        distance_threshold = self.GOAL_THRESHOLD * 0.1
        scale_factor = self.DIRECTION_SCALE_FACTORS[0] if distance_to_goal > distance_threshold else self.DIRECTION_SCALE_FACTORS[1]
    
        return cosine_similarity * scale_factor if cosine_similarity < 0 else 0
    
    def visiting_reward(self, current_pos, chunk):
        return self.VISITING_REWARD if np.linalg.norm(chunk[0] - current_pos) < self.WAYPOINT_THRESHOLD else 0.0

    def path_following_reward(self, current_pos, chunk):
        if chunk is None:
            return 0

        min_dist = float('inf')
        closest_segment_start = None
        segment_idx = 0

        for i in range(len(chunk) - 1):
            p1 = chunk[i]
            p2 = chunk[i + 1]
            segment = p2 - p1
            len_segment = np.linalg.norm(segment)
            if len_segment == 0:
                continue
                
            t = np.dot(current_pos - p1, segment) / (len_segment * len_segment)
            t = max(0, t)
            projection = p1 + t * segment
            dist = np.linalg.norm(current_pos - projection)
            if dist < min_dist:
                min_dist = dist
                closest_segment_start = p1
                segment_idx = i

        if min_dist <= self.PATH_CORRIDOR_WIDTH: # acceptable corridor - positive reward
            reward = self.PATH_FOLLOWING_FACTOR * (1 - min_dist/self.PATH_CORRIDOR_WIDTH)
        elif min_dist <= self.MAX_PATH_DEVIATION: # Outside corridor within max deviation - small penalty
            penalty_ratio = (min_dist - self.PATH_CORRIDOR_WIDTH) / (self.MAX_PATH_DEVIATION - self.PATH_CORRIDOR_WIDTH)
            reward = -self.PATH_FOLLOWING_FACTOR * penalty_ratio
        else: # Far from path - maximum penalty
            reward = self.DEVIATION_PENALTY_FACTOR

        if closest_segment_start is not None: # progress bonus
            max_segments = len(chunk) - 1
            progress_ratio = segment_idx / max_segments if max_segments > 0 else 0
            progress_bonus = 5 * progress_ratio
            reward += progress_bonus

        clipped_reward = np.clip(reward, -10, 10)

        return clipped_reward
    
    def out_of_boundary_penalty(self, current_pos):
        if np.any(current_pos[0] < self.BOUNDARY_LIMITS[0][0]) or np.any(current_pos[0] > self.BOUNDARY_LIMITS[0][1]):
            return self.BOUNDARY_PENALTY
        elif np.any(current_pos[1] < self.BOUNDARY_LIMITS[1][0]) or np.any(current_pos[1] > self.BOUNDARY_LIMITS[1][1]):
            return self.BOUNDARY_PENALTY
        return 0

    def action_smoothness_reward(self, action: np.ndarray) -> float:
        L, theta = action
        reward = 0.2
        
        if self.prev_L is not None and self.prev_theta is not None:
            # Handle L smoothness
            if abs(self.prev_L) < 1e-6:  # If previous L was close to zero
                L_change = abs(L)  # Consider absolute new value
            else:
                # Consider both magnitude and direction changes
                L_change = (abs(L - self.prev_L) / (self.prev_L + 1e-6))
            
            # Handle theta smoothness
            if abs(self.prev_theta) < 1e-6:  # If previous theta was close to zero
                theta_change = abs(theta)  # Consider absolute new value
            else:
                # Consider both magnitude and direction changes
                theta_change = (abs(theta - self.prev_theta) / (self.prev_theta + 1e-6))
            
            # Apply penalties for changes beyond allowable thresholds
            if L_change > self.ALLOWABLE_L_CHANGE:
                # excess_change = L_change - self.ALLOWABLE_L_CHANGE
                L_penalty = self.L_SMOOTHNESS_PENALTY * min(L_change, self.MAX_PENALTY_THRESHOLD)
                reward += L_penalty
            
            if theta_change > self.ALLOWABLE_THETA_CHANGE:
                # excess_change = theta_change - self.ALLOWABLE_THETA_CHANGE
                theta_penalty = self.THETA_SMOOTHNESS_PENALTY * min(theta_change, self.MAX_PENALTY_THRESHOLD)
                reward += theta_penalty
        
        # Update previous values
        self.prev_L = L
        self.prev_theta = theta
        
        return reward