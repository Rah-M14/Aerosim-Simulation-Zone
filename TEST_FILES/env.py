import gymnasium as gym
import numpy as np
import wandb
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle, Arrow, Polygon
# import matplotlib.colors as mcolors
# import matplotlib.transforms as mtransforms
import os
import cv2

from LiDAR import *

from RRTStar import RRTStarPlanner, gen_goal_pose
from Path_Manager import PathManager
from configs import ObservationConfig
from new_reward_manager import RewardManager
from reward_monitor import RewardMonitor

obs_config = ObservationConfig()

class PathFollowingEnv(gym.Env):    
    def __init__(self, image_path: str, algo_run: str, max_episode_steps: int = 1000, chunk_size: int = obs_config.chunk_size, headless: bool = False, enable_reward_monitor: bool = False, enable_wandb: bool = False):
        super().__init__()
        
        self.image_path = image_path
        self.chunk_size = chunk_size
        self.name = algo_run
        self.headless = headless

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
        
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_num = 1
        
        self.agent_theta = 0.0
        self.world_max = np.array([8, 6])
        self.world_limits = np.array([[-8, 8], [-6, 6]])
        self.env_world_limits = np.array([[-10, 10], [-8, 8]])
        self.world_diag = np.linalg.norm(self.world_max)

        # Movement parameters
        self.max_step_size = 1  # Maximum distance the agent can move in one step
        self.max_theta = np.pi  # Maximum angle the agent can move in one step

        # LiDAR Parameters
        self.lidar_specs = [1, 360, 720, 200] # [resolution, FOV, #Rays, distance]

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32
        )
        
        # Obs: [
        #   current_x, current_y,                   # Current position (2)
        #   goal_x, goal_y,                         # Final goal position (2)
        #   next_waypoints (chunk_size * 2),        # Next waypoints in path (chunk_size * 2)
        #   distance_to_goal,                       # Scalar distance to final goal (1)
        #   distance_to_next_waypoint               # Scalar distance to next waypoint (1)
        #   Timestep t                              # Timestep (1)
        # ]

        obs_size = self.start.shape[-1] + self.goal_pos.shape[-1] + (chunk_size * 2) + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.path_manager = PathManager(image_path, chunk_size=chunk_size)
        self.reward_manager = RewardManager(self.agent_theta, monitor=None)
        
        # Initialize reward monitor if enabled
        if enable_reward_monitor:
            self.reward_manager.monitor = RewardMonitor(max_steps=max_episode_steps)
        
        if enable_wandb:
            self.wandb_enabled = True
            wandb.define_metric("RL_step")
            wandb.define_metric("episode_num")
            wandb.define_metric("episode_num", step_metric="RL_step")

            wandb.define_metric("Learning Curve", step_metric="episode_num")
            wandb.define_metric("episode_length", step_metric="episode_num")

            wandb.define_metric("Action_L", step_metric="RL_step")
            wandb.define_metric("Action_Theta", step_metric="RL_step")
            # wandb.define_metric("Total Reward (Step)", step_metric="RL_step")
            wandb.define_metric("Goal Reached", step_metric="RL_step")
            wandb.define_metric("Timeout", step_metric="RL_step")
            wandb.define_metric("Boundary", step_metric="RL_step")
            wandb.define_metric("success_reward", step_metric="RL_step")
            wandb.define_metric("timeout_penalty", step_metric="RL_step")
            wandb.define_metric("boundary_penalty", step_metric="RL_step")
            wandb.define_metric("total_reward", step_metric="RL_step")
            wandb.define_metric("goal_potential", step_metric="RL_step")
            wandb.define_metric("path_potential", step_metric="RL_step")
            wandb.define_metric("progress", step_metric="RL_step")
            wandb.define_metric("path_following", step_metric="RL_step")
            wandb.define_metric("heading", step_metric="RL_step")
            wandb.define_metric("oscillation_penalty", step_metric="RL_step")
        else:
            self.wandb_enabled = False

        if not self.headless:
            self.output_dir = "render_frames"
            os.makedirs(self.output_dir, exist_ok=True)
        
            # Initialize figure for off-screen rendering
            plt.ioff()  # Turn off interactive mode
            self.fig, self.ax = plt.subplots(figsize=(10, 8), facecolor='black')
            self.ax.set_facecolor('black')
            
            # Create named window for OpenCV
            cv2.namedWindow('Path Following Environment - ' + self.name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Path Following Environment - ' + self.name, 800, 600)
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        if not self.headless:
            self.ax.clear()
        
        # Reset reward monitor if it exists
        if hasattr(self.reward_manager, 'monitor'):
            self.reward_manager.monitor.reset()

        if self.wandb_enabled:
            wandb.log({"episode_num": self.episode_num})
            wandb.log({"episode_length": self.episode_length})
            wandb.log({"Learning Curve": self.episode_reward})

        # Reset internal counters
        self.current_step = 0
        self.episode_reward = 0
        self.episode_num += 1

        self.current_pos = self.gen_bot_pos()
        self.prev_pos = self.current_pos
        # self.goal_pos = self.gen_bot_pos()
        self.goal_pos = self.current_pos + np.random.uniform(-3.0, 3.0, size=2)

        # Reset path manager and plan new path
        self.path_manager.reset()
        self.path_manager.plan_new_path(self.current_pos, self.goal_pos)
        self.current_chunk = self.path_manager.get_next_chunk(self.current_pos)

        self.reward_manager.reset()
        
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
        
        if self.wandb_enabled:
            wandb.log({"RL_step": self.global_step})
            wandb.log({"Action_L": action[0]})
            wandb.log({"Action_Theta": action[1]})
        
        reward, reward_components = self.reward_manager.compute_reward(
            current_pos=self.current_pos,
            prev_pos=self.prev_pos,
            goal_pos=self.goal_pos,
            chunk=self.current_chunk,
            world_theta=self.agent_theta
        )
        
        self.episode_reward += reward
        
        done = False
        truncated = False
        
        # Update info dict with reward components
        info = {
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'distance_to_goal': observation[-3],
            'distance_to_next': observation[-2],
            'action_linear': action[0],
            'action_angular': action[1],
            'reward_components': reward_components,
            'success': False,
            'timeout': False,
            'boundary': False,
            'total_reward': reward
        }
        
        if info['distance_to_goal'] < 0.1:
            done = True
            success_reward = self.reward_manager.GOAL_REACHED_REWARD
            reward += success_reward
            info['success'] = True
            reward_components['success_reward'] = success_reward
            wandb.log({"Goal Reached": 1}) if self.wandb_enabled else None
        else:
            wandb.log({"Goal Reached": 0}) if self.wandb_enabled else None
            reward_components['success_reward'] = 0.0
        
        if self.current_step >= self.max_episode_steps:
            truncated = True
            timeout_penalty = self.reward_manager.TIMEOUT_PENALTY
            reward += timeout_penalty
            info['timeout'] = True
            reward_components['timeout_penalty'] = timeout_penalty
            wandb.log({"Timeout": 1}) if self.wandb_enabled else None
        else:
            wandb.log({"Timeout": 0}) if self.wandb_enabled else None
            reward_components['timeout_penalty'] = 0.0

        if self.reward_manager.out_of_boundary_penalty(self.current_pos) < 0.0:
            truncated = True
            boundary_penalty = self.reward_manager.BOUNDARY_PENALTY
            reward += boundary_penalty
            info['boundary'] = True
            reward_components['boundary_penalty'] = boundary_penalty
            wandb.log({"Boundary": 1}) if self.wandb_enabled else None
        else:
            wandb.log({"Boundary": 0}) if self.wandb_enabled else None
            reward_components['boundary_penalty'] = 0.0

        reward_components['total_reward'] = reward

        if self.wandb_enabled:
            wandb.log(reward_components)

        if not self.headless:
            self.render()
        
        return observation, reward, done, truncated, info
    
    def action_to_pos(self, action: np.ndarray) -> np.ndarray:
        L = action[0] * self.max_step_size
        theta = action[1] * self.max_theta
        
        self.agent_theta += theta
        self.agent_theta = ((self.agent_theta + np.pi) % (2*np.pi)) - np.pi
        self.prev_theta = theta

        return np.array([self.current_pos[0] + (L*np.cos(self.agent_theta)), self.current_pos[1] + (L*np.sin(self.agent_theta))])
    
    def _get_observation(self) -> np.ndarray:
        dist_to_goal = np.linalg.norm(self.current_pos - self.goal_pos)
        dist_to_next = np.linalg.norm(self.current_pos - self.current_chunk[0])
        
        if self.current_chunk.shape[0] != self.chunk_size:
            fixed_chunk = np.zeros((self.chunk_size, 2))
            
            actual_points = min(self.current_chunk.shape[0], self.chunk_size)
            fixed_chunk[:actual_points] = self.current_chunk[:actual_points]
            
            if actual_points < self.chunk_size:
                fixed_chunk[actual_points:] = self.current_chunk[-1]
                
            self.current_chunk = fixed_chunk
        
        flat_chunk = (self.current_chunk / self.world_max).flatten()
        
        obs = np.concatenate([
            self.current_pos / self.world_max,               # Current position (2)
            self.goal_pos / self.world_max,                  # Goal position (2)
            flat_chunk,                   # Waypoints (chunk_size * 2)
            np.array([dist_to_goal / self.world_diag]),       # Distance to goal (1)
            np.array([dist_to_next / self.world_diag]),       # Distance to next waypoint (1)
            np.array([self.current_step / self.max_episode_steps])   # Timestep (1)
        ]).astype(np.float32)
        
        expected_size = 2 + 2 + (self.chunk_size * 2) + 3
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
            
            # Get LiDAR points using actual world limits for simulation
            # points = lidar_simulation_from_image(
            #     image_path=self.image_path, 
            #     left_corner=(-10, 7),  # Fixed world limits for LiDAR simulation
            #     right_corner=(10, -7), 
            #     start=self.current_pos, 
            #     end=self.goal_pos, 
            #     resolution=self.lidar_specs[0], 
            #     fov=self.lidar_specs[1], 
            #     num_rays=self.lidar_specs[2], 
            #     max_distance=self.lidar_specs[3]
            # )
            
            # # Draw LiDAR points
            # for point in points:
            #     world_x = point[0]/self.lidar_specs[0] + self.current_pos[0]
            #     world_y = point[1]/self.lidar_specs[0] + self.current_pos[1]
            #     img_x, img_y = world_to_img(world_x, world_y)
            #     cv2.circle(img, (img_x, img_y), 2, (0, 0, 255), -1)  # Red dots for LiDAR points
                
                # # Only draw points within the visible area
                # if -8 <= world_x <= 8 and -6 <= world_y <= 6:
                #     img_x, img_y = world_to_img(world_x, world_y)
                #     if 0 <= img_x < 800 and 0 <= img_y < 600:  # Check if point is within image bounds
                #         cv2.circle(img, (img_x, img_y), 2, (0, 0, 255), -1)  # Red dots for LiDAR points
            
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
            
            if hasattr(self.reward_manager, 'monitor'):
                reward_info = [
                    f"Episode Reward: {self.episode_reward:.2f}",
                    f"Current Reward: {self.reward_manager.monitor.histories['total_reward'][-1]:.2f}" if self.reward_manager.monitor.histories['total_reward'] else "N/A",
                    f"Goal Potential: {self.reward_manager.monitor.histories['goal_potential'][-1]:.2f}" if self.reward_manager.monitor.histories['goal_potential'] else "N/A",
                    f"Path Potential: {self.reward_manager.monitor.histories['path_potential'][-1]:.2f}" if self.reward_manager.monitor.histories['path_potential'] else "N/A",
                    f"Path Following: {self.reward_manager.monitor.histories['path_following'][-1]:.2f}" if self.reward_manager.monitor.histories['path_following'] else "N/A",
                    f"Progress: {self.reward_manager.monitor.histories['progress'][-1]:.2f}" if self.reward_manager.monitor.histories['progress'] else "N/A",
                    f"Heading: {self.reward_manager.monitor.histories['heading'][-1]:.2f}" if self.reward_manager.monitor.histories['heading'] else "N/A",
                    f"Oscillation Penalty: {self.reward_manager.monitor.histories['oscillation_penalty'][-1]:.2f}" if self.reward_manager.monitor.histories['oscillation_penalty'] else "N/A"
                ]
                
                for i, text in enumerate(reward_info):
                    cv2.putText(img, text, (500, 30 + i * 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
            
            # Show the image
            cv2.imshow(f'Path Following Environment - {self.name}', img)
            # cv2.waitKey(1500)
            cv2.waitKey(1)
            
        except Exception as e:
            print(f"Render error: {e}")
            
    def close(self):
        if not self.headless:
            try:
                plt.close(self.fig)
                plt.close('all')
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                
                # Close reward monitor if it exists
                if hasattr(self.reward_manager, 'monitor'):
                    plt.close(self.reward_manager.monitor.fig)
            except Exception as e:
                print(f"Close error: {e}")
        else:
            pass
            
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        if not self.headless:
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

        # return np.array([0.0, 0.0])
        return new_pos