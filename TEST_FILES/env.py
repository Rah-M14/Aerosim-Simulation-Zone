import gymnasium as gym
import numpy as np
import wandb
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image

from LiDAR_Fast import *
from RRTStar import RRTStarPlanner, gen_goal_pose
from Path_Manager import PathManager
from configs import ObservationConfig
from new_reward_manager import RewardManager
from reward_monitor import RewardMonitor

obs_config = ObservationConfig()

class PathFollowingEnv(gym.Env):    
    def __init__(self, image_path: str, algo_run: str, max_episode_steps: int = 1000, chunk_size: int = obs_config.chunk_size, headless: bool = False, 
    enable_reward_monitor: bool = False, enable_wandb: bool = False, render_type: str = "normal"):
        super().__init__()
        
        self.image_path = image_path
        img = np.array(Image.open(self.image_path).convert('L'))  # Convert to grayscale
        self.binary_img = (img > 128).astype(np.uint8)  # Threshold to create a binary map
        self.binary_img = cv2.resize(self.binary_img, (0,0), fx=0.25, fy=0.25)
        
        self.chunk_size = chunk_size
        self.name = algo_run
        self.headless = headless
        self.render_type = render_type

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
        self.world_max = np.array([10, 7])
        self.world_limits = np.array([[-8, 8], [-6, 6]])
        self.env_world_limits = np.array([[-10, 10], [-8, 8]])
        self.world_diag = np.linalg.norm(self.world_max)

        # Movement parameters
        self.max_step_size = 1  # Maximum distance the agent can move in one step
        self.max_theta = 2*np.pi  # Maximum angle the agent can move in one step

        # LiDAR Parameters
        self.lidar_specs = [1, 360, 720, 500] # [resolution, FOV, #Rays, distance]
        self.lidar_bounds = ((-10, 8), (10, -8))  # (left_corner, right_corner)
        self.lidar_thresh = 1.5
        self.lidar_points = None
        self.lidar_dists = None
        self.lidar_mask = None

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32
        )
        
        obs_size_new = 2 + 2 + 1 + 1    # Current pos (2), Goal pos (2), Agent theta (1), Relative theta (1)
        lidar_mask_size = 360           # Number of LiDAR rays (360)

        self.observation_space = spaces.Dict(
            {
                "vector": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(obs_size_new,),
                    dtype=np.float32
                    ),
                "lidar_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(lidar_mask_size,),
                    dtype=np.float32,
                ),
            }
        )

        self.path_manager = PathManager(self.image_path, chunk_size=chunk_size)
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
            wandb.define_metric("Goal Reached", step_metric="RL_step")
            wandb.define_metric("Timeout", step_metric="RL_step")
            wandb.define_metric("Boundary", step_metric="RL_step")
            wandb.define_metric("success_reward", step_metric="RL_step")
            wandb.define_metric("timeout_penalty", step_metric="RL_step")
            wandb.define_metric("boundary_penalty", step_metric="RL_step")
            wandb.define_metric("total_reward", step_metric="RL_step")
            wandb.define_metric("goal_potential", step_metric="RL_step")
            # wandb.define_metric("path_potential", step_metric="RL_step")
            wandb.define_metric("progress", step_metric="RL_step")
            # wandb.define_metric("path_following", step_metric="RL_step")
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
            if self.render_type == "dual":
                cv2.resizeWindow('Path Following Environment - ' + self.name, 800*2, 600)
            else:
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
        # self.current_pos = np.array([0, 0])
        self.prev_pos = self.current_pos
        self.goal_pos = self.gen_bot_pos()
        # self.goal_pos = np.array([7, 7])

        # Reset path manager and plan new path
        # self.path_manager.reset()
        # self.path_manager.plan_new_path(self.current_pos, self.goal_pos)
        # self.current_chunk = self.path_manager.get_next_chunk(self.current_pos)

        self.reward_manager.reset()
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        self.global_step += 1
        self.episode_length += 1

        pos = self.action_to_pos(action)

        self.prev_pos = self.current_pos
        self.current_pos = pos
        # self.current_chunk = self.path_manager.get_next_chunk(self.current_pos)
        
        observation = self._get_observation()

        # print(f"Step: {self.current_step}, Action: {action}, Vector: {observation['vector'].shape}, Lidar Mask: {observation['lidar_mask'].shape}")
        
        if self.wandb_enabled:
            wandb.log({"RL_step": self.global_step})
            wandb.log({"Action_L": action[0]})
            wandb.log({"Action_Theta": action[1]})
        
        reward, reward_components = self.reward_manager.compute_reward(
            current_pos=self.current_pos,
            prev_pos=self.prev_pos,
            goal_pos=self.goal_pos,
            world_theta=self.agent_theta,
            timestep=self.current_step
        )
        
        self.episode_reward += reward
        
        done = False
        truncated = False
        
        # Update info dict with reward components
        info = {
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'Agent_Theta': observation['vector'][-2],
            'Goal_Relative_Theta': observation['vector'][-1],
            'action_linear': action[0],
            'action_angular': action[1],
            'reward_components': reward_components,
            'success': False,
            'timeout': False,
            'boundary': False,
            'total_reward': reward
        }

        goal_dist = np.linalg.norm(self.goal_pos - self.current_pos)
        
        if goal_dist < 0.1:
            done = True
            success_reward = self.reward_manager.GOAL_REACHED_REWARD
            reward = success_reward
            info['success'] = True
            reward_components['success_reward'] = success_reward
            wandb.log({"Goal Reached": 1}) if self.wandb_enabled else None
        else:
            wandb.log({"Goal Reached": 0}) if self.wandb_enabled else None
            reward_components['success_reward'] = 0.0
        
        if self.current_step >= self.max_episode_steps:
            truncated = True
            timeout_penalty = self.reward_manager.TIMEOUT_PENALTY
            reward = timeout_penalty
            info['timeout'] = True
            reward_components['timeout_penalty'] = timeout_penalty
            wandb.log({"Timeout": 1}) if self.wandb_enabled else None
        else:
            wandb.log({"Timeout": 0}) if self.wandb_enabled else None
            reward_components['timeout_penalty'] = 0.0

        if self.reward_manager.out_of_boundary_penalty(self.current_pos) < 0.0:
            truncated = True
            boundary_penalty = self.reward_manager.BOUNDARY_PENALTY
            reward = boundary_penalty
            info['boundary'] = True
            reward_components['boundary_penalty'] = boundary_penalty
            wandb.log({"Boundary": 1}) if self.wandb_enabled else None
        else:
            wandb.log({"Boundary": 0}) if self.wandb_enabled else None
            reward_components['boundary_penalty'] = 0.0

        reward_components['total_reward'] = reward

        if self.wandb_enabled:
            wandb.log(reward_components)

        self.lidar_points, self.lidar_dists = get_lidar_points(
            self.binary_img,
            self.current_pos,
            self.env_world_limits,
            num_rays=360,
            max_range=4.0
        )
        if not self.headless:
            self.render()
        
        return observation, reward, done, truncated, info
    
    def action_to_pos(self, action: np.ndarray) -> np.ndarray:
        L = action[0] * self.max_step_size
        theta = action[1] * self.max_theta
        
        self.agent_theta += theta
        self.agent_theta = ((self.agent_theta + 2*np.pi) % (2*np.pi))
        self.prev_theta = theta

        return np.array([self.current_pos[0] + (L*np.cos(self.agent_theta)), self.current_pos[1] + (L*np.sin(self.agent_theta))])
    
    def _get_observation(self) -> np.ndarray:
        self.lidar_points, self.lidar_dists = get_lidar_points(
            self.binary_img,
            self.current_pos,
            self.env_world_limits,
            num_rays=360,
            max_range=4.0
        )
        self.lidar_mask = get_safe_mask(self.lidar_dists, self.lidar_thresh)
        
        goal_vec = self.goal_pos - self.current_pos
        
        obs = {
            'vector' : np.concatenate([
        self.current_pos / self.world_max,                                                      # Current position (2)
            self.goal_pos / self.world_max,                                                     # Goal position (2)
            np.array([self.agent_theta / (2*np.pi)]),                                           # Agent Theta (1)
            np.array([(np.arctan2(goal_vec[1], goal_vec[0]) - self.agent_theta) / (2*np.pi)])   # Relative Goal Theta (1)
            ]).astype(np.float32), 
            'lidar_mask' : self.lidar_mask                                                      # LiDAR Mask (1,360)
            }
        
        # print(f'observation lidar mask shape: {obs["lidar_mask"].shape}')
        
        expected_v_size = 2 + 2 + 1 + 1
        expected_li_size = 360
        assert obs['vector'].shape[0] == expected_v_size, f"Observation shape mismatch. Expected {expected_v_size}, got {obs['vector'].shape[0]}"
        assert obs['lidar_mask'].shape[0] == expected_li_size, f"Observation shape mismatch. Expected {expected_li_size}, got {obs['lidar_mask'].shape[0]}"
        
        return obs

    def render(self):
        if self.render_type == "dual":
            try:
                # Create two side-by-side windows with fixed dimensions and a gap
                img_width, img_height = 800, 600
                gap_size = 20  # Gap between panels
                combined_img = np.zeros((img_height, img_width * 2 + gap_size, 3), dtype=np.uint8)
                
                # Load and process the world image for the left panel
                world_img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
                if world_img is None:
                    raise ValueError(f"Failed to load image from {self.image_path}")
                
                # Resize and convert to BGR
                world_img = cv2.resize(world_img, (img_width, img_height))
                world_img = cv2.cvtColor(world_img, cv2.COLOR_GRAY2BGR)
                
                # Copy the world image to the left panel
                combined_img[:, :img_width] = world_img
                
                # Fill gap with dark color
                combined_img[:, img_width:img_width+gap_size] = (30, 30, 30)
                
                # Adjust coordinate conversion functions for the right panel
                def world_to_img_left(x, y):
                    # Transform from world coordinates to image coordinates using env limits
                    x_range = self.env_world_limits[0][1] - self.env_world_limits[0][0]
                    y_range = self.env_world_limits[1][1] - self.env_world_limits[1][0]
                    
                    img_x = int(round((x - self.env_world_limits[0][0]) * (img_width / x_range)))
                    img_y = int(round((self.env_world_limits[1][1] - y) * (img_height / y_range)))
                    return img_x, img_y

                def world_to_img_right(x, y):
                    # Transform from world coordinates to image coordinates using env limits (for the right panel)
                    x_range = self.env_world_limits[0][1] - self.env_world_limits[0][0]
                    y_range = self.env_world_limits[1][1] - self.env_world_limits[1][0]
                    
                    img_x = int(round((x - self.env_world_limits[0][0]) * (img_width / x_range))) + img_width + gap_size
                    img_y = int(round((self.env_world_limits[1][1] - y) * (img_height / y_range)))
                    return img_x, img_y

                # Draw grid on right panel using env_world_limits
                x_min, x_max = self.env_world_limits[0]
                y_min, y_max = self.env_world_limits[1]
                
                # Draw vertical grid lines
                for x in range(int(x_min), int(x_max) + 1):
                    x_img = world_to_img_right(x, 0)[0]
                    cv2.line(combined_img, (x_img, 0), (x_img, img_height), (50, 50, 50), 1)
                    
                # Draw horizontal grid lines
                for y in range(int(y_min), int(y_max) + 1):
                    y_img = world_to_img_right(0, y)[1]
                    cv2.line(combined_img, (img_width + gap_size, y_img), 
                            (img_width * 2 + gap_size, y_img), (50, 50, 50), 1)

                # Draw reward monitor if it exists (in bottom right corner)
                if hasattr(self.reward_manager, 'monitor') and self.reward_manager.monitor is not None:
                    monitor_data = self.reward_manager.monitor.get_data()
                    if monitor_data is not None and len(monitor_data) > 1:
                        # Define reward plot area
                        plot_width = 200
                        plot_height = 100
                        plot_margin = 20
                        plot_x = img_width * 2 - plot_width - plot_margin
                        plot_y = img_height - plot_height - plot_margin
                        
                        # Draw plot background
                        cv2.rectangle(combined_img, 
                                    (plot_x, plot_y), 
                                    (plot_x + plot_width, plot_y + plot_height), 
                                    (30, 30, 30), -1)  # Dark gray background
                        cv2.rectangle(combined_img, 
                                    (plot_x, plot_y), 
                                    (plot_x + plot_width, plot_y + plot_height), 
                                    (100, 100, 100), 1)  # Light gray border
                        
                        # Scale reward data to fit plot
                        rewards = np.array(monitor_data)
                        min_reward = np.min(rewards)
                        max_reward = np.max(rewards)
                        reward_range = max_reward - min_reward if max_reward != min_reward else 1
                        
                        # Draw reward curve
                        points = []
                        for i in range(len(rewards)):
                            x = plot_x + int((i / len(rewards)) * plot_width)
                            normalized_reward = (rewards[i] - min_reward) / reward_range
                            y = plot_y + plot_height - int(normalized_reward * plot_height)
                            points.append((x, y))
                        
                        # Draw the curve
                        for i in range(len(points) - 1):
                            cv2.line(combined_img, points[i], points[i + 1], (0, 0, 255), 1)
                        
                        # Add min/max labels
                        font_scale = 0.4
                        cv2.putText(combined_img, f"max: {max_reward:.1f}", 
                                (plot_x + 5, plot_y + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
                        cv2.putText(combined_img, f"min: {min_reward:.1f}", 
                                (plot_x + 5, plot_y + plot_height - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

                # Draw path on both panels
                if self.path_manager.get_full_path() is not None:
                    path = self.path_manager.get_full_path()
                    if len(path) > 1:
                        for i in range(len(path) - 1):
                            # Left panel
                            start_l = world_to_img_left(path[i][0], path[i][1])
                            end_l = world_to_img_left(path[i+1][0], path[i+1][1])
                            cv2.line(combined_img, start_l, end_l, (0, 255, 255), 2)
                            
                            # Right panel
                            start_r = world_to_img_right(path[i][0], path[i][1])
                            end_r = world_to_img_right(path[i+1][0], path[i+1][1])
                            cv2.line(combined_img, start_r, end_r, (0, 255, 255), 2)

                # Draw agent on both panels
                for is_left in [True, False]:
                    world_to_img = world_to_img_left if is_left else world_to_img_right
                    
                    # Get agent position in image coordinates
                    agent_pos = world_to_img(self.current_pos[0], self.current_pos[1])
                    
                    # Triangle parameters
                    triangle_size = 15
                    angle = self.agent_theta
                    
                    # Calculate triangle vertices
                    tip = (
                        int(agent_pos[0] + np.cos(angle) * triangle_size),
                        int(agent_pos[1] - np.sin(angle) * triangle_size)
                    )
                    base_l = (
                        int(agent_pos[0] - np.cos(angle + np.pi/6) * triangle_size),
                        int(agent_pos[1] + np.sin(angle + np.pi/6) * triangle_size)
                    )
                    base_r = (
                        int(agent_pos[0] - np.cos(angle - np.pi/6) * triangle_size),
                        int(agent_pos[1] + np.sin(angle - np.pi/6) * triangle_size)
                    )
                    
                    # Draw yellow triangle body
                    triangle_pts = np.array([tip, base_l, base_r], np.int32)
                    cv2.fillPoly(combined_img, [triangle_pts], (0, 255, 255))  # Yellow fill
                    
                    # Draw red tip
                    tip_size = 5
                    red_tip = (
                        int(agent_pos[0] + np.cos(angle) * triangle_size),
                        int(agent_pos[1] - np.sin(angle) * triangle_size)
                    )
                    cv2.circle(combined_img, red_tip, tip_size, (0, 0, 255), -1)  # Red circle at tip
                    
                    # Draw goal
                    goal_pos = world_to_img(self.goal_pos[0], self.goal_pos[1])
                    cv2.circle(combined_img, goal_pos, 10, (0, 255, 0), -1)  # Green circle

                # Draw LiDAR points on both panels
                if self.lidar_points is not None and len(self.lidar_points) > 0:
                    # Draw rays and points
                    for point in self.lidar_points:
                        # Convert LiDAR points to world coordinates
                        world_x = self.current_pos[0] + point[0]
                        world_y = self.current_pos[1] + point[1]
                        
                        # Draw on left panel
                        left_x, left_y = world_to_img_left(world_x, world_y)
                        robot_left = world_to_img_left(self.current_pos[0], self.current_pos[1])
                        # cv2.line(combined_img, robot_left, (left_x, left_y), (0, 0, 255), 1)  # Red ray
                        cv2.circle(combined_img, (left_x, left_y), 2, (255, 0, 0), -1)  # Blue dot
                        
                        # Draw on right panel
                        right_x, right_y = world_to_img_right(world_x, world_y)
                        robot_right = world_to_img_right(self.current_pos[0], self.current_pos[1])
                        # cv2.line(combined_img, robot_right, (right_x, right_y), (0, 0, 255), 1)  # Red ray
                        cv2.circle(combined_img, (right_x, right_y), 2, (255, 0, 0), -1)  # Blue dot

                # Add text overlay on left panel
                info_text = [
                    f"Episode: {self.episode_num}",
                    f"Step: {self.current_step}",
                    f"Reward: {self.episode_reward:.2f}",
                    f"Distance to Goal: {np.linalg.norm(self.current_pos - self.goal_pos):.2f}",
                    f"Angle: {np.degrees(self.agent_theta):.1f}"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(combined_img, text, (10, 30 + i * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Add reward component text on right panel if monitor is enabled
                if hasattr(self.reward_manager, 'monitor') and self.reward_manager.monitor is not None:
                    reward_text = [
                        "Reward Components:",
                        # f"Success: {self.reward_manager.monitor.histories['success_reward'][-1] if len(self.reward_manager.monitor.histories['success_reward']) > 0 else 0:.2f}",
                        # f"Timeout: {self.reward_manager.monitor.histories['timeout_penalty'][-1] if len(self.reward_manager.monitor.histories['timeout_penalty']) > 0 else 0:.2f}",
                        # f"Boundary: {self.reward_manager.monitor.histories['boundary_penalty'][-1] if len(self.reward_manager.monitor.histories['boundary_penalty']) > 0 else 0:.2f}",
                        f"Goal Potential: {self.reward_manager.monitor.histories['goal_potential'][-1] if len(self.reward_manager.monitor.histories['goal_potential']) > 0 else 0:.2f}",
                        # f"Path Potential: {self.reward_manager.monitor.histories['path_potential'][-1] if len(self.reward_manager.monitor.histories['path_potential']) > 0 else 0:.2f}",
                        f"Progress: {self.reward_manager.monitor.histories['progress'][-1] if len(self.reward_manager.monitor.histories['progress']) > 0 else 0:.2f}",
                        # f"Path Following: {self.reward_manager.monitor.histories['path_following'][-1] if len(self.reward_manager.monitor.histories['path_following']) > 0 else 0:.2f}",
                        f"Heading: {self.reward_manager.monitor.histories['heading'][-1] if len(self.reward_manager.monitor.histories['heading']) > 0 else 0:.2f}",
                        f"Oscillation: {self.reward_manager.monitor.histories['oscillation_penalty'][-1] if len(self.reward_manager.monitor.histories['oscillation_penalty']) > 0 else 0:.2f}",
                        f"Total: {self.reward_manager.monitor.histories['total_reward'][-1] if len(self.reward_manager.monitor.histories['total_reward']) > 0 else 0:.2f}"
                    ]
                    
                    for i, text in enumerate(reward_text):
                        cv2.putText(combined_img, text, 
                                (img_width + 10, 30 + i * 25),  # Positioned on left side of right panel
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Smaller font size

                # Show the combined image
                cv2.imshow(f'Path Following Environment - {self.name}', combined_img)
                cv2.waitKey(1000)
            
            except Exception as e:
                print(f"Render error: {str(e)}")
                import traceback
                traceback.print_exc()
                
        elif self.render_type == "normal":
            try:
                # Create single window with fixed dimensions
                img_width, img_height = 800, 600
                display_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                
                def world_to_img(x, y):
                    # Transform from world coordinates to image coordinates using env limits
                    x_range = self.env_world_limits[0][1] - self.env_world_limits[0][0]
                    y_range = self.env_world_limits[1][1] - self.env_world_limits[1][0]
                    
                    img_x = int(round((x - self.env_world_limits[0][0]) * (img_width / x_range)))
                    img_y = int(round((self.env_world_limits[1][1] - y) * (img_height / y_range)))
                    return img_x, img_y

                # Draw grid using env_world_limits
                x_min, x_max = self.env_world_limits[0]
                y_min, y_max = self.env_world_limits[1]
                
                # Draw vertical grid lines
                for x in range(int(x_min), int(x_max) + 1):
                    x_img = world_to_img(x, 0)[0]
                    cv2.line(display_img, (x_img, 0), (x_img, img_height), (50, 50, 50), 1)
                    
                # Draw horizontal grid lines
                for y in range(int(y_min), int(y_max) + 1):
                    y_img = world_to_img(0, y)[1]
                    cv2.line(display_img, (0, y_img), (img_width, y_img), (50, 50, 50), 1)

                # Draw path
                if self.path_manager.get_full_path() is not None:
                    path = self.path_manager.get_full_path()
                    if len(path) > 1:
                        for i in range(len(path) - 1):
                            start = world_to_img(path[i][0], path[i][1])
                            end = world_to_img(path[i+1][0], path[i+1][1])
                            cv2.line(display_img, start, end, (0, 255, 255), 2)

                # Draw agent
                agent_pos = world_to_img(self.current_pos[0], self.current_pos[1])
                triangle_size = 15
                angle = self.agent_theta
                
                # Calculate triangle vertices
                tip = (
                    int(agent_pos[0] + np.cos(angle) * triangle_size),
                    int(agent_pos[1] - np.sin(angle) * triangle_size)
                )
                base_l = (
                    int(agent_pos[0] - np.cos(angle + np.pi/6) * triangle_size),
                    int(agent_pos[1] + np.sin(angle + np.pi/6) * triangle_size)
                )
                base_r = (
                    int(agent_pos[0] - np.cos(angle - np.pi/6) * triangle_size),
                    int(agent_pos[1] + np.sin(angle - np.pi/6) * triangle_size)
                )
                
                # Draw yellow triangle body
                triangle_pts = np.array([tip, base_l, base_r], np.int32)
                cv2.fillPoly(display_img, [triangle_pts], (0, 255, 255))
                
                # Draw red tip
                tip_size = 5
                red_tip = (
                    int(agent_pos[0] + np.cos(angle) * triangle_size),
                    int(agent_pos[1] - np.sin(angle) * triangle_size)
                )
                cv2.circle(display_img, red_tip, tip_size, (0, 0, 255), -1)
                
                # Draw goal
                goal_pos = world_to_img(self.goal_pos[0], self.goal_pos[1])
                cv2.circle(display_img, goal_pos, 10, (0, 255, 0), -1)

                # Draw episode info (top left)
                episode_info = [
                    f"Episode: {self.episode_num}",
                    f"Step: {self.current_step}",
                    f"Reward: {self.episode_reward:.2f}",
                    f"Distance: {np.linalg.norm(self.current_pos - self.goal_pos):.2f}",
                    f"Angle: {np.degrees(self.agent_theta):.1f}°"
                ]
                
                for i, text in enumerate(episode_info):
                    cv2.putText(display_img, text, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Draw reward components (top right) if monitor exists
                if hasattr(self.reward_manager, 'monitor') and self.reward_manager.monitor is not None:
                    reward_text = [
                        "Reward Components:",
                        f"Goal Potential: {self.reward_manager.monitor.histories['goal_potential'][-1] if len(self.reward_manager.monitor.histories['goal_potential']) > 0 else 0:.2f}",
                        # f"Path Potential: {self.reward_manager.monitor.histories['path_potential'][-1] if len(self.reward_manager.monitor.histories['path_potential']) > 0 else 0:.2f}",
                        f"Progress: {self.reward_manager.monitor.histories['progress'][-1] if len(self.reward_manager.monitor.histories['progress']) > 0 else 0:.2f}",
                        # f"Path Following: {self.reward_manager.monitor.histories['path_following'][-1] if len(self.reward_manager.monitor.histories['path_following']) > 0 else 0:.2f}",
                        f"Heading: {self.reward_manager.monitor.histories['heading'][-1] if len(self.reward_manager.monitor.histories['heading']) > 0 else 0:.2f}",
                        f"Oscillation: {self.reward_manager.monitor.histories['oscillation_penalty'][-1] if len(self.reward_manager.monitor.histories['oscillation_penalty']) > 0 else 0:.2f}",
                        f"Total: {self.reward_manager.monitor.histories['total_reward'][-1] if len(self.reward_manager.monitor.histories['total_reward']) > 0 else 0:.2f}"
                    ]
                    
                    for i, text in enumerate(reward_text):
                        cv2.putText(display_img, text,
                                (img_width - 300, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    # Draw reward monitor plot (bottom right)
                    monitor_data = self.reward_manager.monitor.get_data()
                    if monitor_data is not None and len(monitor_data) > 1:
                        plot_width = 250
                        plot_height = 150
                        plot_margin = 20
                        plot_x = img_width - plot_width - plot_margin
                        plot_y = img_height - plot_height - plot_margin
                        
                        # Draw plot background and border
                        cv2.rectangle(display_img, 
                                    (plot_x, plot_y),
                                    (plot_x + plot_width, plot_y + plot_height),
                                    (30, 30, 30), -1)
                        cv2.rectangle(display_img,
                                    (plot_x, plot_y),
                                    (plot_x + plot_width, plot_y + plot_height),
                                    (100, 100, 100), 1)
                        
                        # Scale and plot reward data
                        rewards = np.array(monitor_data)
                        min_reward = np.min(rewards)
                        max_reward = np.max(rewards)
                        reward_range = max_reward - min_reward if max_reward != min_reward else 1
                        
                        points = []
                        for i in range(len(rewards)):
                            x = plot_x + int((i / len(rewards)) * plot_width)
                            normalized_reward = (rewards[i] - min_reward) / reward_range
                            y = plot_y + plot_height - int(normalized_reward * plot_height)
                            points.append((x, y))
                        
                        # Draw curve
                        for i in range(len(points) - 1):
                            cv2.line(display_img, points[i], points[i + 1], (0, 0, 255), 1)
                        
                        # Add min/max labels
                        cv2.putText(display_img, f"max: {max_reward:.1f}",
                                (plot_x + 5, plot_y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        cv2.putText(display_img, f"min: {min_reward:.1f}",
                                (plot_x + 5, plot_y + plot_height - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Show the image
                cv2.imshow(f'Path Following Environment - {self.name}', display_img)
                cv2.waitKey(1)
            
            except Exception as e:
                print(f"Render error: {str(e)}")
                import traceback
                traceback.print_exc()    
                
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
