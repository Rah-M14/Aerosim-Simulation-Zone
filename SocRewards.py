import numpy as np
from sklearn.cluster import DBSCAN
import logging
import wandb
import os
from datetime import datetime

class SocialReward:
    def __init__(self, log_dir):
        self.ep_reward_dict = {'ep_to_goal_rew': 0, 'ep_close_to_goal_rew': 0, 'ep_cosine_sim_rew': 0, 'ep_close_coll_rew': 0, 'ep_collision_rew': 0, 'ep_boundary_coll_rew': 0, 'ep_current_rew': 0}

        self.current_rew = 0
        self.ep_reward_to_point = 0

        # PATH REWARD PARAMS
        self.to_goal_rew = 0
        self.close_to_goal_rew = 0
        self.cosine_sim_rew = 0

        # SOCIAL REWARD PARAMS
        self.close_coll_rew = 0
        self.collision_rew = 0

        # LIVE REWARD
        self.live_reward = -1

        # SOCIAL THRESHOLDS & WEIGHTS
        self.coll_dist_threshold = 2.0
        self.coll_threshold = 0.3
        self.coll_exp_rew_scale = 2.0
        self.min_cluster_size = 3
        self.cluster_eps = 0.5
        self.angle_weight = 0.5

        # BOUNDARY THRESHOLDS
        self.x_limits = (-9,9)
        self.y_limits = (-7,7)
        self.boundary_threshold = 0.1
        self.boundary_coll_penalty = -10000

        # GOAL & TERMINAL REWARDS & THRESHOLDS
        self.timeout_penalty = -10000
        self.goal_reward = 1000000
        self.goal_dist_thresold = 0.3

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
            
            # Check if we can write to the file
            with open(log_file, 'w') as f:
                f.write("Initializing log file\n")
            
            # Create a logger
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            
            # Create file handler which logs even debug messages
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            
            # Create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            
            # Add the handlers to the logger
            logger.addHandler(fh)
            
            logger.info("Logging initialized successfully")
            return logger
        except Exception as e:
            print(f"Error setting up logging: {e}")
            # Fallback to console logging if file logging fails
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            logger.warning(f"Falling back to console logging due to error: {e}")
            return logger

    def get_total_reward(self):
        self.logger.info(f"Total reward: {self.to_point_rew}")
        return self.to_point_rew

    def reset_episode_rewards(self):
        self.ep_reward_dict = {
            'ep_to_goal_rew': 0,
            'ep_close_to_goal_rew': 0,
            'ep_cosine_sim_rew': 0,
            'ep_close_coll_rew': 0,
            'ep_collision_rew': 0,
            'ep_current_rew': 0
        }
        self.episode_length = 0
        self.logger.info("Episode rewards reset")
    

    # TOTAL REWARD COMPUTATION
    def compute_reward(self, prev_bot_pos, cur_bot_pos, goal_pos, lidar_data=None):
        reward_dict = {'to_goal_rew': 0, 'close_to_goal_rew': 0, 'cosine_sim_rew': 0, 'close_coll_rew': 0, 'collision_rew': 0, 'current_rew': 0}
        self.episode_length += 1

        self.to_goal_rew = self.towards_goal_reward(prev_bot_pos, cur_bot_pos, goal_pos)
        self.close_to_goal_rew = self.close_to_goal_reward(cur_bot_pos, goal_pos)
        self.cosine_sim_rew = self.cosine_similarity_reward(prev_bot_pos, cur_bot_pos, goal_pos)

        self.close_coll_rew = self.close_collision_reward(lidar_data)
        self.collision_rew = -100 if self.check_collision(cur_bot_pos, lidar_data) else 0

        self.current_rew = (
            self.to_goal_rew + self.close_to_goal_rew + self.cosine_sim_rew + self.close_coll_rew + self.collision_rew + self.live_reward)

        reward_dict.update({
            'to_goal_rew': self.to_goal_rew,
            'close_to_goal_rew': self.close_to_goal_rew,
            'cosine_sim_rew': self.cosine_sim_rew,
            'close_coll_rew': self.close_coll_rew,
            'collision_rew': self.collision_rew,
            'current_rew': self.current_rew
        })

        if self.collision_rew != 0:
            self.logger.info(f"Collision detected. Penalty: {self.collision_rew}")

        self.logger.info(f"Reward computed: {reward_dict}")
        self.logger.info(f"Current Computed reward: {self.current_rew}")
        wandb.log(reward_dict)

        self.to_point_rew, self.ep_reward_dict = self._update_reward(reward_dict)
        self.logger.info(f"Updated episode rewards: {self.ep_reward_dict}")
        return self.current_rew, self.to_point_rew, reward_dict, self.ep_reward_dict
    
    def _update_reward(self, reward_dict):
        self.ep_reward_dict['ep_to_goal_rew'] += reward_dict['to_goal_rew']
        self.ep_reward_dict['ep_close_to_goal_rew'] += reward_dict['close_to_goal_rew']
        self.ep_reward_dict['ep_cosine_sim_rew'] += reward_dict['cosine_sim_rew']
        self.ep_reward_dict['ep_close_coll_rew'] += reward_dict['close_coll_rew']
        self.ep_reward_dict['ep_collision_rew'] += reward_dict['collision_rew']
        self.ep_reward_dict['ep_current_rew'] += reward_dict['current_rew']

        self.logger.info(f"Updated episode rewards: {self.ep_reward_dict}")
        wandb.log(self.ep_reward_dict)
        return self.ep_reward_dict['ep_current_rew'], self.ep_reward_dict


    # PATH REWARDS
    def towards_goal_reward(self, prev_bot_pos, cur_bot_pos, goal_pos):
        prev_dist_to_goal = np.linalg.norm(goal_pos - prev_bot_pos)
        cur_dist_to_goal = np.linalg.norm(goal_pos - cur_bot_pos)
        self.to_goal_rew = prev_dist_to_goal - cur_dist_to_goal
        # if self.to_goal_rew > 0:
        #     self.to_goal_rew = 5
        # else:
        #     self.to_goal_rew = -10
        return self.to_goal_rew
    
    def close_to_goal_reward(self, cur_bot_pos, goal_pos):
        dist_to_goal = np.linalg.norm(goal_pos - cur_bot_pos)
        if dist_to_goal < self.goal_dist_thresold:
            self.close_to_goal_rew = self.goal_reward
            return self.close_to_goal_rew
        return 0
    
    def cosine_similarity_reward(self, prev_bot_pos, cur_bot_pos, goal_pos):
        bot_vec = cur_bot_pos - prev_bot_pos
        bot_goal_vec = goal_pos - prev_bot_pos
        self.cosine_sim_rew = np.dot(bot_vec, bot_goal_vec) / (np.linalg.norm(bot_vec) * np.linalg.norm(bot_goal_vec))
        return self.cosine_sim_rew


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
        return total_reward
    
    def check_collision(self, agent_position, lidar_data):
        if lidar_data is None or len(lidar_data) == 0:
            self.logger.warning("Lidar data is empty or None for collision check")
            return False
        distances = np.linalg.norm(lidar_data[:, :2] - agent_position[:2], axis=1)
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
        x, y = cur_bot_pos[:2]
        x_min, x_max = self.x_limits
        y_min, y_max = self.y_limits

        x_collision = abs(x - x_min) < self.boundary_threshold or abs(x - x_max) < self.boundary_threshold
        y_collision = abs(y - y_min) < self.boundary_threshold or abs(y - y_max) < self.boundary_threshold

        return x_collision or y_collision
    
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