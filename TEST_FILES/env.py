import gymnasium as gym
import numpy as np
import wandb
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for off-screen rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow, Polygon
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
import os
import cv2

from RRTStar import RRTStarPlanner, gen_goal_pose
from Path_Manager import PathManager
from Reward_Manager import RewardManager
from configs import ObservationConfig

obs_config = ObservationConfig()

class PathFollowingEnv(gym.Env):    
    def __init__(self, image_path: str, algo: str, max_episode_steps: int = 1000, chunk_size: int = obs_config.chunk_size):
        super().__init__()
        
        self.path_manager = PathManager(image_path, chunk_size=chunk_size)
        self.reward_manager = RewardManager()
        self.chunk_size = chunk_size
        self.algo = algo


        self.start = np.zeros(2)
        self.goal_pos = np.zeros(2)
        self.current_chunk = None
        self.current_pos = None
        self.prev_pos = None
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.global_step = 0
        
        self.dist_to_goal = 0.0
        self.dist_to_next = 0.0
        
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_num = 0
        
        self.agent_theta = 0.0


        # Movement parameters
        self.max_step_size = 1  # Maximum distance the agent can move in one step
        self.max_theta = np.pi / 9  # Maximum angle the agent can move in one step
        # Actions: [L, theta] - continuous values between -1 and 1
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Obs: [
        #   current_x, current_y,                   # Current position (2)
        #   goal_x, goal_y,                         # Final goal position (2)
        #   next_waypoints (chunk_size * 2),        # Next waypoints in path (chunk_size * 2)
        #   distance_to_goal,                       # Scalar distance to final goal (1)
        #   distance_to_next_waypoint               # Scalar distance to next waypoint (1)
        # ]
        obs_size = self.start.shape[-1] + self.goal_pos.shape[-1] + (chunk_size * 2) + 2
        print(f"obs_size: {obs_size}")
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )        
        
        # Create output directory for frames if it doesn't exist
        self.output_dir = "render_frames"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize figure for off-screen rendering
        plt.ioff()  # Turn off interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 8), facecolor='black')
        self.ax.set_facecolor('black')
        
        # Create named window for OpenCV
        cv2.namedWindow('Path Following Environment - ' + self.algo, cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Path Following Environment - ' + self.algo, 800, 600)
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Clear the figure but don't close it
        self.ax.clear()
        
        # Reset internal counters
        self.current_step = 0
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_num += 1

        self.coll_term = 1
        self.timeout_tem = -1

        # Generate new start and goal positions
        self.current_pos = self.gen_bot_pos()
        self.prev_pos = self.current_pos
        self.goal_pos = self.gen_bot_pos()

        # Initialize agent_theta if it hasn't been set
        self.agent_theta = 0.0
        
        # Reset path manager and plan new path
        self.path_manager.reset()
        self.path_manager.plan_new_path(self.current_pos, self.goal_pos)

        self.current_chunk = self.path_manager.get_next_chunk(self.current_pos)
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        self.global_step += 1
        self.episode_length += 1
        
        new_pos = self.action_to_pos(action)
        self.prev_pos = self.current_pos
        self.current_pos = new_pos
        self.current_chunk = self.path_manager.get_next_chunk(self.current_pos)
        
        observation = self._get_observation()
        reward = self.reward_manager.compute_reward(
            current_pos=self.current_pos,
            prev_pos=self.prev_pos,
            goal_pos=self.goal_pos,
            chunk=self.current_chunk,
            action=action
        )
        
        self.episode_reward += reward
        
        done = False
        truncated = False
        info = {
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'distance_to_goal': observation[-2],
            'distance_to_next': observation[-1],
            'action_linear': action[0],
            'action_angular': action[1]
        }
        
        if info['distance_to_goal'] < 0.1:
            done = True
            reward += self.reward_manager.SUCCESS_REWARD
            info['success'] = True
        
        if self.current_step >= self.max_episode_steps:
            truncated = True
            wandb.log({"Ep_Termination": self.timeout_tem, 'episode_num' : self.episode_num})
            reward += self.reward_manager.TIMEOUT_PENALTY
            info['timeout'] = True
        elif self.reward_manager.out_of_boundary_penalty(self.current_pos) < 0.0:
            truncated = True
            wandb.log({"Ep_Termination": self.coll_term, 'episode_num' : self.episode_num})
            reward += self.reward_manager.BOUNDARY_PENALTY
            info['boundary'] = True
        wandb.log({"Ep_Termination": 0, 'episode_num' : self.episode_num})

        # Log episode data when done
        if done or truncated:
            wandb.log({'episode': self.episode_num, 'episode_num' : self.episode_num})
            wandb.log({'episode_reward': self.episode_reward, 'episode_num' : self.episode_num})
            wandb.log({'episode_length': self.episode_length, 'episode_num' : self.episode_num})
        
        # Log step data
        wandb.log({'step_reward': reward, 'global_step' : self.global_step})
        wandb.log({'distance_to_goal': info['distance_to_goal'], 'global_step' : self.global_step})
        wandb.log({'distance_to_next': info['distance_to_next'], 'global_step' : self.global_step})
        wandb.log({'action_linear': info['action_linear'], 'global_step' : self.global_step})
        wandb.log({'action_angular': info['action_angular'], 'global_step' : self.global_step})
        wandb.log({'global_step': self.global_step})
        
        self.render()
        
        return observation, reward, done, truncated, info
    
    def action_to_pos(self, action: np.ndarray) -> np.ndarray:
        L = action[0] * self.max_step_size
        theta = action[1] * self.max_theta
        
        self.agent_theta += theta

        if self.agent_theta > 0 and self.agent_theta < np.pi:
            self.agent_theta = self.agent_theta
        else:
            self.agent_theta = -(np.pi - self.agent_theta % (np.pi))
        
        self.prev_theta = theta

        return np.array([self.current_pos[0] + (L*np.cos(self.agent_theta)), self.current_pos[1] + (L*np.sin(self.agent_theta))])
    
    def _get_observation(self) -> np.ndarray:
        dist_to_goal = np.linalg.norm(self.current_pos - self.goal_pos)
        dist_to_next = np.linalg.norm(self.current_pos - self.current_chunk[0])
        
        # Ensure chunk is the correct size
        if self.current_chunk.shape[0] != self.chunk_size:
            # Create a properly sized chunk array
            fixed_chunk = np.zeros((self.chunk_size, 2))
            
            # Fill with actual waypoints as much as possible
            actual_points = min(self.current_chunk.shape[0], self.chunk_size)
            fixed_chunk[:actual_points] = self.current_chunk[:actual_points]
            
            # If we have fewer points than chunk_size, fill the rest with the last valid point
            if actual_points < self.chunk_size:
                fixed_chunk[actual_points:] = self.current_chunk[-1]
                
            self.current_chunk = fixed_chunk
        
        flat_chunk = self.current_chunk.flatten()
        
        obs = np.concatenate([
            self.current_pos,           # Current position (2)
            self.goal_pos,              # Goal position (2)
            flat_chunk,                 # Waypoints (chunk_size * 2)
            np.array([dist_to_goal]),   # Distance to goal (1)
            np.array([dist_to_next])    # Distance to next waypoint (1)
        ]).astype(np.float32)
        
        expected_size = 2 + 2 + (self.chunk_size * 2) + 2
        assert obs.shape[0] == expected_size, f"Observation shape mismatch. Expected {expected_size}, got {obs.shape[0]}"
        
        return obs
    
    def render(self):
        try:
            img = np.zeros((600, 800, 3), dtype=np.uint8)
            
            # Convert world coordinates to image coordinates
            def world_to_img(x, y):
                img_x = int((x + 8) * 50)  # Scale and shift x
                img_y = int((6 - y) * 50)  # Scale and shift y
                return img_x, img_y
            
            # Draw grid
            for x in range(-8, 9):
                x_img = int((x + 8) * 50)
                cv2.line(img, (x_img, 0), (x_img, 600), (50, 50, 50), 1)
            for y in range(-6, 7):
                y_img = int((6 - y) * 50)
                cv2.line(img, (0, y_img), (800, y_img), (50, 50, 50), 1)
            
            # Draw path
            if self.path_manager.get_full_path() is not None:
                path = np.array(self.path_manager.get_full_path())
                if len(path) > 1:
                    for i in range(len(path) - 1):
                        pt1 = world_to_img(path[i][0], path[i][1])
                        pt2 = world_to_img(path[i+1][0], path[i+1][1])
                        cv2.line(img, pt1, pt2, (150, 150, 150), 1)
            
            # Draw current chunk
            if self.current_chunk is not None and len(self.current_chunk) > 1:
                for i in range(len(self.current_chunk) - 1):
                    pt1 = world_to_img(self.current_chunk[i][0], self.current_chunk[i][1])
                    pt2 = world_to_img(self.current_chunk[i+1][0], self.current_chunk[i+1][1])
                    cv2.line(img, pt1, pt2, (0, 165, 255), 2)
            
            # Draw agent as triangle with red tip
            agent_pos = world_to_img(self.current_pos[0], self.current_pos[1])
            triangle_size = 15
            
            # Calculate triangle vertices
            angle = self.agent_theta  # The angle in radians
            tip = (
                int(agent_pos[0] + np.cos(angle) * triangle_size),
                int(agent_pos[1] - np.sin(angle) * triangle_size)
            )
            base_l = (
                int(agent_pos[0] - np.cos(angle) * triangle_size - np.sin(angle) * triangle_size),
                int(agent_pos[1] + np.sin(angle) * triangle_size - np.cos(angle) * triangle_size)
            )
            base_r = (
                int(agent_pos[0] - np.cos(angle) * triangle_size + np.sin(angle) * triangle_size),
                int(agent_pos[1] + np.sin(angle) * triangle_size + np.cos(angle) * triangle_size)
            )
            
            # Draw main triangle body (cyan)
            triangle_pts = np.array([tip, base_l, base_r], np.int32)
            cv2.fillPoly(img, [triangle_pts], (255, 255, 0))  # Cyan body
            
            # Draw red tip
            tip_length = triangle_size
            red_tip = (
                int(agent_pos[0] + np.cos(angle) * triangle_size),
                int(agent_pos[1] - np.sin(angle) * triangle_size)
            )
            red_base_l = (
                int(agent_pos[0] + np.cos(angle) * (triangle_size - tip_length) - np.sin(angle) * (tip_length//2)),
                int(agent_pos[1] - np.sin(angle) * (triangle_size - tip_length) - np.cos(angle) * (tip_length//2))
            )
            red_base_r = (
                int(agent_pos[0] + np.cos(angle) * (triangle_size - tip_length) + np.sin(angle) * (tip_length//2)),
                int(agent_pos[1] - np.sin(angle) * (triangle_size - tip_length) + np.cos(angle) * (tip_length//2))
            )
            red_triangle = np.array([red_tip, red_base_l, red_base_r], np.int32)
            cv2.fillPoly(img, [red_triangle], (0, 0, 255))  # Red tip
            
            # Draw goal
            goal_pos = world_to_img(self.goal_pos[0], self.goal_pos[1])
            cv2.circle(img, goal_pos, 15, (0, 255, 0), -1)
            
            # Add text information
            info_text = [
                f"Episode: {self.episode_num}",
                f"Step: {self.current_step}",
                f"Reward: {self.episode_reward:.2f}",
                f"G_Distance: {np.linalg.norm(self.current_pos - self.goal_pos):.2f}",
                f"N_Distance: {np.linalg.norm(self.current_pos - self.current_chunk[0]):.2f}",
                f"Angle: {np.degrees(self.agent_theta):.1f}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(img, text, (10, 30 + i * 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Show the image
            cv2.imshow(f'Path Following Environment - {self.algo}', img)
            cv2.waitKey(1)
            
        except Exception as e:
            print(f"Render error: {e}")
            
    def close(self):
        """Close all windows safely."""
        try:
            plt.close(self.fig)
            plt.close('all')
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # This helps ensure windows are properly closed
        except Exception as e:
            print(f"Close error: {e}")
            
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.close()
                
    def gen_bot_pos(self):
        new_pos = np.array(
            [
                np.random.choice(
                    list(
                        set([x for x in np.linspace(-7.5, 7.6, 10000)])
                        - set(
                            y
                            for y in np.append(
                                np.linspace(-2.6, -1.7, 900),
                                np.append(
                                    np.linspace(-0.8, 0.4, 1200),
                                    np.append(
                                        np.linspace(1.5, 2.4, 900),
                                        np.linspace(3.4, 4.6, 1200),
                                    ),
                                ),
                            )
                        )
                    )
                ),
                np.random.choice(
                    list(
                        set([x for x in np.linspace(-5.5, 5.6, 14000)])
                        - set(
                            y
                            for y in np.append(
                                np.linspace(-1.5, 2.5, 1000),
                                np.linspace(-2.5, -5.6, 3100),
                            )
                        )
                    )
                )
            ]
        )
        return new_pos