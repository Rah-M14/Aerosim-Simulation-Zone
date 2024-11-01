import numpy as np
from sklearn.cluster import DBSCAN
import logging
import wandb
import os
from datetime import datetime

class SocialReward:
    def __init__(self, log_dir):
        # self.ep_reward_dict = {'ep_close_to_goal_rew': 0, 'ep_cosine_sim_rew': 0, 'ep_dist_to_goal_rew': 0, 'ep_close_coll_rew': 0, 'ep_collision_rew': 0, 'ep_boundary_coll_rew': 0, 'ep_current_rew': 0}
        
        self.ep_reward_dict = {
            'ep_cosine_sim_rew': 0,
            'ep_dist_to_goal_rew': 0,
            'ep_smooth_crash_rew': 0,
            # 'ep_hard_crash_rew': 0,
            'ep_close_coll_rew': 0,
            'ep_collision_rew': 0,
            'ep_goal_reached_rew': 0,
            'ep_current_rew': 0
        }

        self.current_rew = 0
        self.ep_reward_to_point = 0

        # PATH REWARD PARAMS
        self.to_goal_rew = 0
        self.close_to_goal_rew = 0
        self.cosine_sim_rew = 0
        self.dist_to_goal_rew = 0
        self.to_goal_factor = 50
        self.cosine_sim_factor = 500
        self.dist_goal_factor = 0.5

        # SOCIAL REWARD PARAMS
        self.close_coll_rew = 0
        self.collision_rew = 0

        # LIVE REWARD
        self.live_reward = -1

        # SOCIAL THRESHOLDS & WEIGHTS
        self.coll_dist_threshold = 1.0
        self.coll_threshold = 0.2
        self.coll_exp_rew_scale = 10
        self.collision_penalty = -100
        self.min_cluster_size = 3
        self.cluster_eps = 0.5
        self.angle_weight = 0.5

        # BOUNDARY THRESHOLDS
        self.x_limits = (-9,9)
        self.y_limits = (-7,7)
        self.boundary_threshold = 0.5
        self.boundary_coll_penalty = -50

        # GOAL & TERMINAL REWARDS & THRESHOLDS
        # self.timeout_penalty = -1000
        self.timeout_penalty = 0
        self.goal_reward = 10000
        self.goal_dist_threshold = 2
        self.close_goal_dist_threshold = 0.5

        # CURRICULUM REWARDS
        self.curriculum_level = 4
        self.level_rewards = {
            1: {"goal": 20000, "step": -0.1},
            2: {"goal": 20000, "step": -0.2},
            3: {"goal": 20000, "step": -0.3},
            4: {"goal": 20000, "step": -0.5}
        }

        # NEW PARAMETERS
        # self.max_distance_per_step = 0.1  # Adjust based on your environment
        self.max_distance = 10.0  # Maximum possible distance in the environment
        self.boundary_limits = (-9.5, 9.5, -6.5, 6.5)  # (x_min, x_max, y_min, y_max)
        self.direction_scale_factors = [5, 0.1]
        self.distance_scale_factor = 100
        self.smooth_crash_threshold = 1.5

        # LOGGING
        self.episode_length = 0
        self.log_dir = os.path.join(log_dir, "Reward_Logs")
        self.logger = self.setup_logging()
        print(f"Logger initialized: {self.logger}")
        print(f"Logger handlers: {self.logger.handlers}")
        self.logger.info("This is a test log message from __init__")

    def setup_logging(self):
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.log_dir, f"social_reward_{timestamp}.log")
            
            with open(log_file, 'w') as f:
                f.write("Initializing log file\n")
            
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            
            logger.addHandler(fh)
            
            logger.info("Logging initialized successfully")
            return logger

        except Exception as e:
            print(f"Error setting up logging: {e}")
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            logger.warning(f"Falling back to console logging due to error: {e}")
            return logger

    def log_episode_rewards(self, ep_count):
        if ep_count > 0 and self.episode_length > 0:
            avg_rewards = {f"avg_{k}": v / self.episode_length for k, v in self.ep_reward_dict.items()}
            avg_rewards['episode_length'] = self.episode_length
            wandb.log(avg_rewards)
            self.logger.info(f"Average episode rewards: {avg_rewards}")
        else:
            self.logger.warning("Attempted to log rewards for an episode with zero length")

    def end_episode(self, ep_count):
        self.log_episode_rewards(ep_count)
        self.reset_episode_rewards()

    # CURRICULUM UPDATE
    def update_curriculum_level(self, level):
        self.curriculum_level = level
        self.logger.info(f"Reward system updated to curriculum level {self.curriculum_level}")
        wandb.log({"reward_curriculum_level": self.curriculum_level})

    def get_total_reward(self):
        self.logger.info(f"Total reward: {self.to_point_rew}")
        return self.to_point_rew

    def reset_episode_rewards(self):
        self.ep_reward_dict = {
            'ep_cosine_sim_rew': 0,
            'ep_dist_to_goal_rew': 0,
            'ep_smooth_crash_rew': 0,
            # 'ep_hard_crash_rew': 0,
            'ep_close_coll_rew': 0,
            'ep_collision_rew': 0,
            'ep_goal_reached_rew': 0,
            'ep_current_rew': 0
        }
        self.episode_length = 0
        self.logger.info("Episode rewards reset")

    # TOTAL REWARD COMPUTATION
    def compute_reward(self, prev_bot_pos, cur_bot_pos, goal_pos, lidar_data=None):
        reward_dict = {'cosine_sim_rew': 0, 
                       'dist_to_goal_rew': 0, 
                       'smooth_crash_rew': 0, 
                    #    'hard_crash_rew': 0, 
                       'close_coll_rew': 0, 
                       'collision_rew': 0, 
                       'goal_reached_rew': 0, 
                       'current_rew': 0}
        self.episode_length += 1

        if self.check_goal_reached(cur_bot_pos, goal_pos):
            reward_dict['goal_reached_rew'] = self.level_rewards[self.curriculum_level]["goal"]
            self.current_rew = reward_dict['goal_reached_rew']
            self.logger.info(f"Goal reached! Reward: {self.goal_reward}")
        # elif self.check_boundary(cur_bot_pos):
        #     reward_dict['hard_crash_rew'] = self.hard_crash_penalty()
        #     self.current_rew = reward_dict['hard_crash_rew']
        #     self.logger.info(f"Hard Crash! Reward: {self.current_rew}")
        else:
            direction_reward = self.reward_for_direction(cur_bot_pos, prev_bot_pos, goal_pos)
            scaled_direction_reward = self.scale_direction_reward(direction_reward, cur_bot_pos, goal_pos)
            reward_dict['cosine_sim_rew'] = scaled_direction_reward

            distance_reward = self.reward_for_distance(cur_bot_pos, prev_bot_pos, goal_pos)
            scaled_distance_reward = self.scale_distance_reward(distance_reward, cur_bot_pos, goal_pos)
            reward_dict['dist_to_goal_rew'] = scaled_distance_reward

            smooth_crash_penalty = self.smooth_crash_penalty(cur_bot_pos)
            reward_dict['smooth_crash_rew'] = smooth_crash_penalty

            # reward_dict['close_coll_rew'] = self.close_collision_reward(lidar_data)
            reward_dict['close_coll_rew'] = 0
            # reward_dict['collision_rew'] = self.collision_rew if self.check_collision(cur_bot_pos, lidar_data) else 0
            reward_dict['collision_rew'] = 0

            self.current_rew = scaled_direction_reward + scaled_distance_reward + smooth_crash_penalty + reward_dict['close_coll_rew'] + reward_dict['collision_rew'] + self.live_reward 

        reward_dict['current_rew'] = self.current_rew

        self.logger.info(f"Reward computed: {reward_dict}")
        wandb.log(reward_dict)

        self.to_point_rew, self.ep_reward_dict = self._update_reward(reward_dict)
        return self.current_rew, self.to_point_rew, reward_dict, self.ep_reward_dict


    def hard_crash_penalty(self):
        """Return a hard penalty for crashing."""
        return self.boundary_coll_penalty

    def reward_for_direction(self, cur_bot_pos, prev_bot_pos, goal):
        """Compute direction reward based on cosine similarity."""
        vector_to_goal = np.array(goal) - np.array(cur_bot_pos)
        vector_action = np.array(cur_bot_pos) - np.array(prev_bot_pos)
        cosine_similarity = np.dot(vector_to_goal, vector_action) / (np.linalg.norm(vector_to_goal) * np.linalg.norm(vector_action) + 1e-9)
        return cosine_similarity

    def scale_direction_reward(self, cosine_similarity, cur_bot_pos, goal):
        """Diminish the importance of the direction reward when closer to the goal."""
        distance_to_goal = np.linalg.norm(np.array(goal) - np.array(cur_bot_pos))
        distance_threshold = self.max_distance * 0.1
        scale_factor = self.direction_scale_factors[0] if distance_to_goal > distance_threshold else self.direction_scale_factors[1]
        return cosine_similarity * scale_factor

    def reward_for_distance(self, cur_bot_pos, prev_bot_pos, goal):
        """Compute hybrid distance reward."""
        curr_distance_to_goal = np.linalg.norm(np.array(goal) - np.array(cur_bot_pos))
        prev_distance_to_goal = np.linalg.norm(np.array(goal) - np.array(prev_bot_pos))
        return prev_distance_to_goal - curr_distance_to_goal

    def scale_distance_reward(self, distance_reward, cur_bot_pos, goal):
        """Reward larger steps when far from the goal and smaller, precise steps when close."""
        distance_to_goal = np.linalg.norm(np.array(goal) - np.array(cur_bot_pos))
        scaling_factor_smoothing = np.log1p(distance_to_goal)
        scaled_distance_reward = scaling_factor_smoothing * (distance_reward) * self.distance_scale_factor
        return np.clip(scaled_distance_reward, -5, 5)

    def smooth_crash_penalty(self, cur_bot_pos):
        """Compute a smooth crash penalty based on the distance to the nearest boundary."""
        x, y = cur_bot_pos
        x_min, x_max, y_min, y_max = self.boundary_limits
        distances = [x - x_min, x_max - x, y - y_min, y_max - y]
        min_distance = min(distances)
        if min_distance < self.smooth_crash_threshold:
            return -50 * (self.smooth_crash_threshold - min_distance) / self.smooth_crash_threshold
        return 0

    def check_boundary(self, cur_bot_pos):
        """Check if the agent is outside the boundary limits."""
        x, y = cur_bot_pos
        x_min, x_max, y_min, y_max = self.boundary_limits
        return not (x_min <= x <= x_max and y_min <= y <= y_max)
    
    def _update_reward(self, reward_dict):
        self.ep_reward_dict['ep_cosine_sim_rew'] += reward_dict['cosine_sim_rew']
        self.ep_reward_dict['ep_dist_to_goal_rew'] += reward_dict['dist_to_goal_rew']
        # self.ep_reward_dict['ep_hard_crash_rew'] += reward_dict['hard_crash_rew']
        self.ep_reward_dict['ep_smooth_crash_rew'] += reward_dict['smooth_crash_rew']
        self.ep_reward_dict['ep_close_coll_rew'] += reward_dict['close_coll_rew']
        self.ep_reward_dict['ep_collision_rew'] += reward_dict['collision_rew']
        self.ep_reward_dict['ep_goal_reached_rew'] += reward_dict['goal_reached_rew']
        self.ep_reward_dict['ep_current_rew'] += reward_dict['current_rew']

        self.logger.info(f"Updated episode rewards: {self.ep_reward_dict}")
        wandb.log(self.ep_reward_dict)
        return self.ep_reward_dict['ep_current_rew'], self.ep_reward_dict
    
    # PATH CHECKS
    def check_goal_reached(self, cur_bot_pos, goal_pos):
        dist_to_goal = np.linalg.norm(goal_pos - cur_bot_pos)
        goal_reached = dist_to_goal < self.close_goal_dist_threshold
        return goal_reached

    # # TOTAL REWARD COMPUTATION
    # def compute_reward(self, prev_bot_pos, cur_bot_pos, goal_pos, lidar_data=None):
    #     reward_dict = {'close_to_goal_rew': 0, 'cosine_sim_rew': 0, 'dist_to_goal_rew': 0, 'close_coll_rew': 0, 'collision_rew': 0, 'current_rew': 0}
    #     self.episode_length += 1

    #     # self.to_goal_rew = self.towards_goal_reward(prev_bot_pos, cur_bot_pos, goal_pos)
    #     self.close_to_goal_rew = self.close_to_goal_reward(cur_bot_pos, goal_pos)
    #     # done, self.close_to_goal_rew = self.check_goal_reached(cur_bot_pos, goal_pos)
    #     self.cosine_sim_rew = self.cosine_similarity_reward(prev_bot_pos, cur_bot_pos, goal_pos)
    #     self.dist_to_goal_rew = self.dist_to_goal(cur_bot_pos, goal_pos)

    #     self.close_coll_rew = self.close_collision_reward(lidar_data)
    #     # self.close_coll_rew = 0
    #     self.collision_rew = self.collision_penalty if self.check_collision(cur_bot_pos, lidar_data) else 0
    #     # self.collision_rew = 0

    #     self.current_rew = (self.close_to_goal_rew + self.cosine_sim_rew + self.dist_to_goal_rew + self.close_coll_rew + self.collision_rew + self.live_reward)
    #     self.current_rew += self.level_rewards[self.curriculum_level]["step"]

    #     reward_dict.update({
    #         # 'to_goal_rew': self.to_goal_rew,
    #         'close_to_goal_rew': self.close_to_goal_rew,
    #         'cosine_sim_rew': self.cosine_sim_rew,
    #         'dist_to_goal_rew': self.dist_to_goal_rew,
    #         'close_coll_rew': self.close_coll_rew,episode
        
    #     reward_dict.update({'current_rew': self.current_rew})

    #     if self.collision_rew != 0:
    #         self.logger.info(f"Collision detected. Penalty: {self.collision_rew}")

    #     self.logger.info(f"Reward computed: {reward_dict}")
    #     self.logger.info(f"Current Computed reward: {self.current_rew}")
    #     wandb.log(reward_dict)

    #     self.to_point_rew, self.ep_reward_dict = self._update_reward(reward_dict)
    #     self.logger.info(f"Updated episode rewards: {self.ep_reward_dict}")
    #     return self.current_rew, self.to_point_rew, reward_dict, self.ep_reward_dict
    
    # def _update_reward(self, reward_dict):
    #     # self.ep_reward_dict['ep_to_goal_rew'] += reward_dict['to_goal_rew']
    #     self.ep_reward_dict['ep_close_to_goal_rew'] += reward_dict['close_to_goal_rew']
    #     self.ep_reward_dict['ep_cosine_sim_rew'] += reward_dict['cosine_sim_rew']
    #     self.ep_reward_dict['ep_dist_to_goal_rew'] += reward_dict['dist_to_goal_rew']
    #     self.ep_reward_dict['ep_close_coll_rew'] += reward_dict['close_coll_rew']
    #     self.ep_reward_dict['ep_collision_rew'] += reward_dict['collision_rew']
    #     self.ep_reward_dict['ep_current_rew'] += reward_dict['current_rew']

    #     self.logger.info(f"Updated episode rewards: {self.ep_reward_dict}")
    #     wandb.log(self.ep_reward_dict)
    #     return self.ep_reward_dict['ep_current_rew'], self.ep_reward_dict


    # PATH REWARDS
    # def towards_goal_reward(self, prev_bot_pos, cur_bot_pos, goal_pos):
    #     prev_dist_to_goal = np.linalg.norm(goal_pos - prev_bot_pos)
    #     cur_dist_to_goal = np.linalg.norm(goal_pos - cur_bot_pos)
    #     self.to_goal_rew = self.to_goal_factor * (prev_dist_to_goal - cur_dist_to_goal)
    #     if self.to_goal_rew <= 0.2 :
    #         self.to_goal_rew = -2
    #     return self.to_goal_rew
    
    # def close_to_goal_reward(self, cur_bot_pos, goal_pos):
    #     dist_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(cur_bot_pos))
    #     if dist_to_goal < self.close_goal_dist_threshold:
    #         self.close_to_goal_rew = self.level_rewards[self.curriculum_level]["goal"]
    #         return self.close_to_goal_rew
    #     return 0

    # def dist_to_goal(self, cur_bot_pos, goal_pos):
    #     parabloic_factor = 10
    #     log_factor = 2
    #     log_scaling_factor = 200

    #     cur_dist_to_goal = np.linalg.norm(goal_pos - cur_bot_pos)
    #     if cur_dist_to_goal < self.goal_dist_threshold:
    #         self.dist_to_goal_rew = np.log(cur_dist_to_goal*log_factor)*log_scaling_factor
    #     else:
    #         self.dist_to_goal_rew = np.log(cur_dist_to_goal*log_factor)*log_scaling_factor - parabloic_factor*(cur_dist_to_goal**2)
    #     return self.dist_to_goal_rew
    
    # def cosine_similarity_reward(self, prev_bot_pos, cur_bot_pos, goal_pos):
    #     bot_vec = np.array(cur_bot_pos) - np.array(prev_bot_pos)
    #     bot_goal_vec = np.array(goal_pos) - np.array(prev_bot_pos)
        
    #     bot_norm = np.linalg.norm(bot_vec)
    #     bot_goal_norm = np.linalg.norm(bot_goal_vec)
        
    #     if bot_norm == 0 or bot_goal_norm == 0:
    #         self.cosine_sim_rew = 0
    #     else:
    #         cosine_sim = np.dot(bot_vec, bot_goal_vec) / (bot_norm * bot_goal_norm)
    #         self.cosine_sim_rew = self.cosine_sim_factor * cosine_sim
    #         if self.cosine_sim_rew > 0:
    #             self.cosine_sim_rew += 100

    #     return self.cosine_sim_rew

    # SOCIAL REWARDS
    def close_collision_reward(self, lidar_data):
        self.logger.info("--- Computing Close Collision Reward ---")

        if lidar_data is None or len(lidar_data) == 0:
            self.logger.warning("Lidar data is empty or None")
            wandb.log({"close_collision_reward": 0})
            return 0

        xy_data = lidar_data[:, :2]
        distances = np.linalg.norm(xy_data, axis=1)
        angles = np.arctan2(xy_data[:, 1], xy_data[:, 0])

        log_msg = f"Distance range: min={np.min(distances):.2f}, max={np.max(distances):.2f}\n"
        log_msg += f"Angle range: min={np.min(angles):.2f}, max={np.max(angles):.2f}"
        self.logger.info(log_msg)

        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.min_cluster_size).fit(xy_data)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.logger.info(f"Number of clusters detected: {n_clusters}")
        

        total_reward = 0
        for label in set(labels):
            if label == -1:
                continue
            
            cluster_mask = (labels == label)
            cluster_distances = distances[cluster_mask]
            cluster_angles = angles[cluster_mask]
            
            close_mask = (cluster_distances < self.coll_dist_threshold)
            if np.any(close_mask):
                close_distances = cluster_distances[close_mask]
                close_angles = cluster_angles[close_mask]
                
                dist_reward = (-np.sum(np.exp(self.coll_exp_rew_scale * (self.coll_dist_threshold - close_distances)))) / np.sum(cluster_mask)
                
                angle_factor = np.cos(close_angles)  # cos(0) = 1 (front), cos(±pi/2) = 0 (sides)
                angle_reward = -np.sum(np.abs(angle_factor) * np.exp(self.coll_exp_rew_scale * (self.coll_dist_threshold - close_distances))) / np.sum(cluster_mask)
                
                cluster_reward = (1 - self.angle_weight) * dist_reward + self.angle_weight * angle_reward
                total_reward += cluster_reward

                log_msg = f"Cluster {label}:\n"
                log_msg += f"  Points in cluster: {np.sum(cluster_mask)}\n"
                log_msg += f"  Close points: {np.sum(close_mask)}\n"
                log_msg += f"  Distance-based reward: {dist_reward:.2f}\n"
                log_msg += f"  Angle-based reward: {angle_reward:.2f}\n"
                log_msg += f"  Combined cluster reward: {cluster_reward:.2f}"
                self.logger.info(log_msg)
            else:
                self.logger.info(f"Cluster {label}: No close points")

        self.logger.info(f"Total close collision reward: {total_reward:.2f}")
        # return total_reward
        return total_reward
    
    def check_collision(self, agent_position, lidar_data):
        if lidar_data is None or len(lidar_data) == 0:
            self.logger.warning("Lidar data is empty or None for collision check")
            return False
        distances = np.linalg.norm(lidar_data[:, :2] - agent_position, axis=1)
        min_distance = np.min(distances)
        collision = np.any(distances < self.coll_threshold)

        log_msg = f"Agent position: {agent_position}\n"
        log_msg += f"Minimum distance to obstacle: {min_distance:.2f}\n"
        log_msg += f"Collision detected: {collision}"
        self.logger.info(log_msg)
        wandb.log({"min_distance_to_obstacle": min_distance})
        return np.any(distances < self.coll_threshold)


    # BOUNDARY REWARDS
    def check_boundary_collision(self, cur_bot_pos):
        x, y = cur_bot_pos
        x_min, x_max = self.x_limits
        y_min, y_max = self.y_limits

        x_collision = abs(x - x_min) < self.boundary_threshold or abs(x - x_max) < self.boundary_threshold
        y_collision = abs(y - y_min) < self.boundary_threshold or abs(y - y_max) < self.boundary_threshold

        x_out = x < x_min or x > x_max
        y_out = y < y_min or y > y_max

        return x_collision or y_collision or x_out or y_out