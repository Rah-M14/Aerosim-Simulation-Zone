import numpy as np
from sklearn.cluster import DBSCAN
import logging
import wandb
import os
from datetime import datetime

from configs import RewardConfig

class SocNavManager:
    def __init__(self, components, method, log_dir):

        self.components = [comp.lower() for comp in components] #['path', 'social', 'boundary']
        self.method = method.lower() #'negative' or 'symmetric'

        self.reward_config = RewardConfig()
        
        self.ep_reward_dict = {
            'ep_cosine_sim_rew': 0,
            'ep_dist_to_goal_rew': 0,
            'ep_smooth_crash_rew': 0,
            'ep_close_coll_rew': 0,
            'ep_current_rew': 0
        }

        # REWARD Inits
        self.cosine_sim_rew = 0
        self.dist_to_goal_rew = 0
        self.smooth_crash_rew = 0
        self.close_coll_rew = 0
        self.current_rew = 0

        self.live_reward = -1
        self.timeout_penalty = -20
        self.goal_reward = 200
        self.boundary_coll_penalty = -10

        # SOCIAL THRESHOLDS & WEIGHTS
        self.coll_dist_threshold = self.reward_config.coll_dist_threshold
        self.coll_exp_rew_scale = self.reward_config.coll_exp_rew_scale
        self.min_cluster_size = self.reward_config.min_cluster_size
        self.cluster_eps = self.reward_config.cluster_eps
        self.angle_weight = self.reward_config.angle_weight

        # BOUNDARY THRESHOLDS
        self.x_limits = self.reward_config.x_limits
        self.y_limits = self.reward_config.y_limits
        self.boundary_threshold = self.reward_config.boundary_threshold
        self.close_goal_dist_threshold = self.reward_config.close_goal_dist_threshold

        # NEW PARAMETERS
        self.max_distance = self.reward_config.max_distance
        self.boundary_limits = self.reward_config.boundary_limits
        self.direction_scale_factors = self.reward_config.direction_scale_factors
        self.distance_scale_factor = self.reward_config.distance_scale_factor
        self.smooth_crash_threshold = self.reward_config.smooth_crash_threshold

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
        wandb.define_metric("ep_count")
        if ep_count > 0 and self.episode_length > 0:
            avg_rewards = {f"avg_{k}": v / self.episode_length for k, v in self.ep_reward_dict.items()}
            avg_rewards['episode_length'] = self.episode_length
            wandb.log(avg_rewards)
            wandb.log({"Learning_Curve": self.ep_reward_dict['ep_current_rew'], "ep_count": ep_count})
            self.logger.info(f"Average episode rewards: {avg_rewards}")
        else:
            self.logger.warning("Attempted to log rewards for an episode with zero length")

    def end_episode(self, ep_count):
        self.log_episode_rewards(ep_count)
        self.reset_episode_rewards()

    # CURRICULUM UPDATE
    # def update_curriculum_level(self, level):
    #     self.curriculum_level = level
    #     self.logger.info(f"Reward system updated to curriculum level {self.curriculum_level}")
    #     # wandb.log({"reward_curriculum_level": self.curriculum_level})

    def get_total_reward(self):
        self.logger.info(f"Total reward: {self.to_point_rew}")
        return self.to_point_rew

    def reset_episode_rewards(self):
        self.ep_reward_dict = {
            'ep_cosine_sim_rew': 0,
            'ep_dist_to_goal_rew': 0,
            'ep_smooth_crash_rew': 0,
            'ep_close_coll_rew': 0,
            'ep_current_rew': 0
        }
        self.episode_length = 0
        self.logger.info("Episode rewards reset")

    # TOTAL REWARD COMPUTATION
    def compute_reward(self, prev_bot_pos, cur_bot_pos, goal_pos, lidar_data=None):
        reward_dict = {'cosine_sim_rew': 0, 
                       'dist_to_goal_rew': 0, 
                       'smooth_crash_rew': 0, 
                       'close_coll_rew': 0, 
                       'current_rew': 0}
        self.episode_length += 1

        # if self.check_goal_reached(cur_bot_pos, goal_pos):
        #     reward_dict['goal_reached_rew'] = self.goal_reward
        #     self.current_rew = reward_dict['goal_reached_rew']
        #     self.logger.info(f"Goal reached! Reward: {self.goal_reward}")
        # else:
        direction_reward = self.reward_for_direction(cur_bot_pos, prev_bot_pos, goal_pos)
        scaled_direction_reward = self.scale_direction_reward(direction_reward, cur_bot_pos, goal_pos)
        reward_dict['cosine_sim_rew'] = scaled_direction_reward

        distance_reward = self.reward_for_distance(cur_bot_pos, prev_bot_pos, goal_pos)
        scaled_distance_reward = self.scale_distance_reward(distance_reward, cur_bot_pos, goal_pos)
        reward_dict['dist_to_goal_rew'] = scaled_distance_reward

        smooth_crash_penalty = self.smooth_crash_penalty(cur_bot_pos)
        reward_dict['smooth_crash_rew'] = smooth_crash_penalty

        if "social" in self.components:
            reward_dict['close_coll_rew'] = self.close_collision_reward(lidar_data)
        else:
            reward_dict['close_coll_rew'] = 0

        self.current_rew = reward_dict['cosine_sim_rew'] + reward_dict['dist_to_goal_rew'] + reward_dict['smooth_crash_rew'] + reward_dict['close_coll_rew'] + self.live_reward 

        reward_dict['current_rew'] = self.current_rew

        self.logger.info(f"Reward computed: {reward_dict}")

        # self.to_point_rew, self.ep_reward_dict = self._update_ep_reward(reward_dict)
        return self.current_rew, reward_dict

    def reward_for_direction(self, cur_bot_pos, prev_bot_pos, goal):
        """Compute direction reward based on cosine similarity."""
        vector_to_goal = np.array(goal) - np.array(prev_bot_pos)
        vector_action = np.array(cur_bot_pos) - np.array(prev_bot_pos)
        cosine_similarity = np.dot(vector_to_goal, vector_action) / (np.linalg.norm(vector_to_goal) * np.linalg.norm(vector_action) + 1e-9)
        return cosine_similarity

    def scale_direction_reward(self, cosine_similarity, cur_bot_pos, goal):
        """Diminish the importance of the direction reward when closer to the goal."""
        distance_to_goal = np.linalg.norm(np.array(goal) - np.array(cur_bot_pos))
        distance_threshold = self.max_distance * 0.1
        # scale_factor = self.direction_scale_factors[0] if distance_to_goal > distance_threshold else self.direction_scale_factors[1]
        scale_factor = 1
        if self.method == "negative":
            return cosine_similarity * scale_factor if cosine_similarity < 0 else 0
        else:
            return cosine_similarity * scale_factor
        
    def reward_for_distance(self, cur_bot_pos, prev_bot_pos, goal):
        """Compute hybrid distance reward."""
        curr_distance_to_goal = np.linalg.norm(np.array(goal) - np.array(cur_bot_pos))
        wandb.log({"curr_distance_to_goal": curr_distance_to_goal})
        prev_distance_to_goal = np.linalg.norm(np.array(goal) - np.array(prev_bot_pos))
        dist_diff = prev_distance_to_goal - curr_distance_to_goal
        wandb.log({"dist_difference": dist_diff})
        return dist_diff

    def scale_distance_reward(self, distance_reward, cur_bot_pos, goal):
        """Reward larger steps when far from the goal and smaller, precise steps when close."""
        distance_to_goal = np.linalg.norm(np.array(goal) - np.array(cur_bot_pos))
        scaling_factor_smoothing = np.log1p(distance_to_goal)
        scaled_distance_reward = scaling_factor_smoothing * (distance_reward) * self.distance_scale_factor
        if self.method == "negative":
            # return np.clip(scaled_distance_reward, -5, 5) if distance_reward < 0 else 0
            return distance_reward if distance_reward < 0 else 0
        else:
            return distance_reward

    def smooth_crash_penalty(self, cur_bot_pos):
        """Compute a smooth crash penalty based on the distance to the nearest boundary."""
        x, y = cur_bot_pos
        x_min, x_max, y_min, y_max = self.boundary_limits
        distances = [x - x_min, x_max - x, y - y_min, y_max - y]
        min_distance = min(distances)
        return -2 if min_distance < self.smooth_crash_threshold else 0
        # return -50 * (self.smooth_crash_threshold - min_distance) / self.smooth_crash_threshold

    def check_boundary(self, cur_bot_pos):
        """Check if the agent is outside the boundary limits."""
        x, y = cur_bot_pos
        x_min, x_max, y_min, y_max = self.boundary_limits
        return not (x_min <= x <= x_max and y_min <= y <= y_max)
    
    def update_ep_rewards(self, reward_dict):
        self.ep_reward_dict['ep_cosine_sim_rew'] += reward_dict['cosine_sim_rew']
        self.ep_reward_dict['ep_dist_to_goal_rew'] += reward_dict['dist_to_goal_rew']
        self.ep_reward_dict['ep_smooth_crash_rew'] += reward_dict['smooth_crash_rew']
        self.ep_reward_dict['ep_close_coll_rew'] += reward_dict['close_coll_rew']
        self.ep_reward_dict['ep_current_rew'] += reward_dict['current_rew']

        self.logger.info(f"Updated episode rewards: {self.ep_reward_dict}")
        return self.ep_reward_dict['ep_current_rew'], self.ep_reward_dict
    
    # PATH CHECKS
    def check_goal_reached(self, cur_bot_pos, goal_pos):
        dist_to_goal = np.linalg.norm(goal_pos - cur_bot_pos)
        goal_reached = dist_to_goal < self.close_goal_dist_threshold
        return goal_reached

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
                self.logger.info(f"Average Distance in Cluster {label} : {np.mean(close_distances)}")
                close_angles = cluster_angles[close_mask]
                
                dist_reward = (-np.sum(np.exp(self.coll_exp_rew_scale * (self.coll_dist_threshold - close_distances)))) / np.sum(cluster_mask)
                
                angle_factor = np.cos(close_angles)  # cos(0) = 1 (front), cos(±pi/2) = 0 (sides)
                angle_reward = -np.sum(np.abs(angle_factor) * np.exp(self.coll_exp_rew_scale * (self.coll_dist_threshold - close_distances)) * 1e-3) / np.sum(cluster_mask)

                total_reward += angle_reward

                log_msg = f"Cluster {label}:\n"
                log_msg += f"  Points in cluster: {np.sum(cluster_mask)}\n"
                log_msg += f"  Close points: {np.sum(close_mask)}\n"
                log_msg += f"  Distance-based reward: {dist_reward:.2f}\n"
                log_msg += f"  Angle-based reward: {angle_reward:.2f}\n"
                # log_msg += f"  Combined cluster reward: {cluster_reward:.2f}"
                self.logger.info(log_msg)
            else:
                self.logger.info(f"Cluster {label}: No close points")

        self.logger.info(f"Total close collision reward: {total_reward:.2f}")
        # return total_reward
        return total_reward
    
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