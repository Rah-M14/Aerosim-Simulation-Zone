import argparse
import sys
import os
import glob
import time
import shutil
from datetime import datetime
import carb
import json
import logging
import wandb

from stable_baselines3 import PPO, SAC
# from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
# from stable_baselines3.sac import MlpPolicy as SACMlpPolicy
from stable_baselines3.sac import MultiInputPolicy as SACMultiPolicy
from stable_baselines3.ppo import MultiInputPolicy as PPOMultiPolicy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
th.cuda.empty_cache()
th.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.cuda.amp as amp

import gym
from gym import spaces
import numpy as np

class SocEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(
        self,
        algo,
        botname,
        headless,
        state_normalize,
        logdir,
        mlp_context=120,
        img_context=30,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=1000,
        seed=0,
    ) -> None:
        
        from isaacsim import SimulationApp

        CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": headless, "renderer": "RayTracedLighting"}

        # parser = argparse.ArgumentParser("Usd Load sample")
        # parser.add_argument(
        #     "--usd_path", type=str, help="Path to usd file, relative to our default assets folder", required=True
        # )
        # parser.add_argument("--headless", default=True, action="store_true", help="Run stage headless")
        # parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")

        # args, unknown = parser.parse_known_args()
        # CONFIG["headless"] = args.headless
        self.kit = SimulationApp(launch_config=CONFIG)
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        
        import omni
        import matplotlib.pyplot as plt
        # from gymnasium import spaces
        import torch as th
        import random
        from ultralytics import YOLO
        from collections import deque
        from stable_baselines3 import PPO

        from omni.isaac.core import World, SimulationContext
        # from omni.isaac.core.scenes import Scene

        # from omni.isaac.core.utils import stage
        from omni.isaac.core.utils.extensions import enable_extension
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        import omni.isaac.core.utils.numpy.rotations as rot_utils
        # from omni.isaac.core.utils.stage import add_reference_to_stage
        # from omni.isaac.core.utils.prims import create_prim
        # from omni.isaac.core.utils.viewports import set_camera_view

        from omni.isaac.core.utils.stage import is_stage_loading
        # import omni.anim.navigation.core as nav
        import omni.usd

        # from omni.isaac.core.tasks.base_task import BaseTask

        import omni.kit.viewport.utility
        from omni.isaac.nucleus import get_assets_root_path, is_file
        # import omni.replicator.core as rep

        EXTENSIONS_PEOPLE = [
            'omni.anim.people', 
            'omni.anim.navigation.bundle',
            'omni.anim.navigation.core',
            'omni.anim.timeline',
            'omni.anim.graph.bundle', 
            'omni.anim.graph.core', 
            'omni.anim.graph.ui',
            'omni.anim.retarget.bundle', 
            'omni.anim.retarget.core',
            'omni.anim.retarget.ui', 
            'omni.kit.scripting',
            'omni.graph.io',
            'omni.anim.curve.core',
        ]

        for ext_people in EXTENSIONS_PEOPLE:
            enable_extension(ext_people)

        # import omni.anim.navigation.core as nav
        self.kit.update()

        from New_RL_Bot_Control import RLBotController, RLBotAct
        from New_RL_Bot import RLBot
        from Adv_SocRewards import SocialReward
        from Mod_Pegasus_App import PegasusApp
        from LiDAR_Feed import get_licam_image

        self.assets_root_path = get_assets_root_path()
        if self.assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            self.kit.close()
            sys.exit()

        current_world_usd = "omniverse://localhost/Projects/SIMS/PEOPLE_SIMS/New_Core.usd"
        usd_path = self.assets_root_path + current_world_usd

        try:
            result = is_file(usd_path)
        except:
            result = False

        if result:
            omni.usd.get_context().open_stage(usd_path)
            self.stage = omni.usd.get_context().get_stage()
            # self.scene = UsdPhysics.Scene.Define(self.stage, "/physicsScene")
        else:
            carb.log_error(
                f"the usd path {usd_path} could not be opened, please make sure that {current_world_usd} is a valid usd file in {self.assets_root_path}"
            )
            self.kit.close()
            sys.exit()
        self.kit.update()
        self.kit.update()

        #print("Loading stage...")

        while is_stage_loading():
            self.kit.update()
        #print("Loading Complete")

        self.world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        self.timeline = omni.timeline.get_timeline_interface() 
        self.world.initialize_physics()
        #print("Waiting Room with Physics initialization in the World!")

        enable_extension("omni.isaac.debug_draw")

        self.botname = botname
        self.bot = RLBot(simulation_app=self.kit, world=self.world, timeline=self.timeline, assets_root_path=self.assets_root_path, botname=self.botname)
        self.controller = RLBotController(botname=self.botname)
        self.act = RLBotAct(self.bot.rl_bot, self.controller, n_steps=5)
        print("Bot Initialised!")
        inav = omni.anim.navigation.core.acquire_interface()
        print("Navmesh Created!") 
        self.people = PegasusApp(self.world, self.stage, self.kit, self.timeline)
        print("People initialised!")
        self.kit.update()

        # WORLD PARAMETERS

        self.world_limits = np.array([20.0, 14.0])
        self.world_min_max = np.array([-10, -7, 10, 7]) # (min_x, min_y, max_x, max_y)

        self.carter_vel_limits = np.array([7.0, 7.0])
        self.carter_ang_vel_limits = np.array([np.pi, np.pi])
        self.state_normalize = state_normalize
        self.timestep = 0

        self.frame_num = 0

        # BOT PARAMETERS
        self.bot.rl_bot.start_pose = None
        self.bot.rl_bot.goal_pose = None
        self.max_velocity = 1.0
        self.max_angular_velocity = np.pi * 3

        self.yolo_model = YOLO("yolov9t.pt")

        # RL PARAMETERS
        self.seed(seed)

        mlp_zeros = np.concatenate([np.zeros(2), np.zeros(4), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(1)])
        self.mlp_context_frame_length = mlp_context
        self.img_context_frame_length = img_context
        self.mlp_context_frame = deque([mlp_zeros for _ in range(self.mlp_context_frame_length)], maxlen=self.mlp_context_frame_length)
        self.img_context_frame = deque([np.zeros((3, 64, 64)) for _ in range(self.img_context_frame_length)], maxlen=self.img_context_frame_length)
        
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([2.5, 1.0]), shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'vector': spaces.Box(low=-np.inf, high=np.inf, shape=(self.mlp_context_frame_length, 13), dtype=np.float32),
            # 'vector': spaces.Box(low=-np.inf, high=np.inf, shape=(self.mlp_context_frame_length, 12), dtype=np.float32),
            'image': spaces.Box(low=0, high=1, shape=(self.img_context_frame_length, 3, 64, 64), dtype=np.float32),
            # 'camera_data': spaces.Box(low=0, high=255, shape=(255, 255, 3), dtype=np.uint8)
        })

        self.ep_steps = 0   
        self.episode_count = 0
        self.info = {}

        self.algo = algo

        self.logdir = os.path.join(logdir, self.algo.upper())
        self.env_log_dir = os.path.join(self.logdir, "Env_Logs")
        self.logger = self.setup_logging()
        print(f"Logger initialized: {self.logger}")
        # print(f"Logger handlers: {self.logger.handlers}")
        self.logger.info("This is a test log message from __init__")

        # REWARD PARAMETERS
        self.reward_manager = SocialReward(self.logdir)
        self.max_episode_length = max_episode_length

        # CURRICULUM APPROACH PARAMETERS5, -np.pi
        self.curriculum_level = 4
        self.level_thresholds = {1: 3.0, 2: 7.0, 3: 11.0, 4: float('inf')}
        self.level_success_rate = {1: 0, 2: 0, 3: 0, 4: 0}
        self.level_episodes = {1: 0, 2: 0, 3: 0, 4: 0}
        self.success_threshold = 0.85
        self.episodes_per_level = 100
        self.reward_manager.update_curriculum_level(self.curriculum_level)

        # wandb.init(project="SocNav_Omni", name="environment_logging")

    def setup_logging(self):
        try:
            os.makedirs(self.env_log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.env_log_dir, f"social_env_{timestamp}.log")
            
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

    def reset(self):
        # print("For some reason reset is being called")
        self.episode_count += 1
        self.reward_manager.end_episode(self.episode_count)
        self.kit.update()
        self.bot.bot_reset()
        self.world.reset(True)
        self.timestep = np.array([0.0])

        self.bot.rl_bot.start_pose = self.bot.rl_bot.get_world_pose()
        print(f"Start pose : {self.bot.rl_bot.start_pose}")
        # self.bot.rl_bot.goal_pose = np.array([1.0, -4.0, 0.0])
        self.bot.rl_bot.goal_pose = self._gen_goal_pose()
        print(f"Goal Pose: {self.bot.rl_bot.goal_pose}")

        state_goal_dist = np.linalg.norm(self.bot.rl_bot.goal_pose[:2] - self.bot.rl_bot.start_pose[0][:2])

        # print(f"Start pose : {self.bot.rl_bot.start_pose[0][:2]}")
        # print(f"Goal Pose: {self.bot.rl_bot.goal_pose[:2]}")
        # print(f"State to goal Diist : {state_goal_dist}")
        print(f"Current curriculum level: {self.curriculum_level}")
        self.kit.update()

        observations, _, _, _ = self.get_observations()
        # observations['vector'] = np.concatenate(observations['vector'])
        self.ep_steps = 0

        self.logger.info(f"Episode {self.episode_count} started")
        wandb.log({"episode": self.episode_count})
        return observations

    def get_observations(self):
        self.world.render()
        bot_world_pos, bot_world_ori = self.bot.rl_bot.get_world_pose()
        bot_lin_vel = self.bot.rl_bot.get_linear_velocity()
        bot_ang_vel = self.bot.rl_bot.get_angular_velocity()
        goal_world_pos = self.bot.rl_bot.goal_pose
        timestep = self.timestep
        if self.state_normalize == True:
            # mlp_states = [bot_world_pos[:2]/self.world_limits,
            #                     bot_world_ori / np.linalg.norm(bot_world_ori),
            #                     bot_lin_vel[:2]/self.carter_vel_limits,
            #                     bot_ang_vel[:2]/self.carter_ang_vel_limits,
            #                     goal_world_pos[:2]/self.world_limits]
            mlp_states = [bot_world_pos[:2]/self.world_limits,
                                bot_world_ori / np.linalg.norm(bot_world_ori),
                                bot_lin_vel[:2],
                                bot_ang_vel[:2],
                                goal_world_pos[:2]/self.world_limits,
                                timestep/self.max_episode_length]
        else:
            mlp_states = [bot_world_pos[:2],
                                bot_world_ori,
                                bot_lin_vel[:2],
                                bot_ang_vel[:2],
                                goal_world_pos[:2],
                                timestep]
            
        self.mlp_context_frame.append(np.concatenate(mlp_states))
        mlp_combined_context = self.get_mlp_combined_context()
        
        lidar_data = self.bot.get_denoised_lidar_data()
        camera_data = self.bot.rl_bot_camera.get_rgba()[:, :, :3]
        if lidar_data.size == 0:
            lidar_data = None
            # lidar_image = np.zeros((3, 128, 128))
            licam_image = np.zeros((3, 64, 64))
            img_combined_context = th.zeros((self.img_context_frame_length, 3, 64, 64))
        else:
            # lidar_image = self.get_lidar_image(lidar_data, mlp_states)
            # self.img_context_frame.append(lidar_image)
            li_cam_image = self.get_li_cam_image(lidar_data, camera_data, mlp_states, self.yolo_model, self.world_min_max, project_camera=True, image_size=64)
            self.img_context_frame.append(li_cam_image)
            img_combined_context = self.get_img_combined_context()

        return {'vector': mlp_combined_context, 'image': img_combined_context}, lidar_data, camera_data, mlp_states
    
    def get_mlp_combined_context(self):
        return th.tensor(np.array(self.mlp_context_frame, dtype=np.float32))

    def get_img_combined_context(self):
        return th.tensor(np.array(self.img_context_frame, dtype=np.float32))
    
    def get_li_cam_image(self, lidar_data, camera_image, mlp_obs, yolo_model, world_min_max, project_camera=False, image_size=64):
        from LiDAR_Feed import get_licam_image
        return get_licam_image(lidar_data, camera_image, mlp_obs, yolo_model, world_min_max, project_camera, image_size)
    
    # def get_mlp_combined_context(self):
    #     total_context = []
    #     for context in self.mlp_context_frame:
    #         total_context.append(np.concatenate(context))
    #     return th.tensor(np.array(total_context, dtype=np.float32))

    # def get_img_combined_context(self):
    #     total_context = []
    #     for context in self.img_context_frame:
    #         total_context.append(context)
    #     return th.tensor(np.array(total_context, dtype=np.float32))
    
    # def de_normalize_states(self, observation):
    #     return [observation[0] * self.world_limits,
    #                             observation[1],
    #                             observation[2] * self.carter_vel_limits,
    #                             observation[3] * self.carter_ang_vel_limits,
    #                             observation[4] * self.world_limits]

    def de_normalize_states(self, observation):
        return [observation[0] * self.world_limits,
                                observation[1],
                                observation[2],
                                observation[3],
                                observation[4] * self.world_limits]

    def step(self, action): # action = [linear_velocity, angular_velocity] both between [-1,1]
        prev_bot_pos, _ = self.bot.rl_bot.get_world_pose()
        self.timestep += 1.0

        if action[0] > 3:
            action[0] = 3
        lin_vel = action[0] * self.max_velocity
        ang_vel = action[1] * self.max_angular_velocity
        # for _ in range(self._skip_frame):
        self.act.move_bot(vals=np.array([lin_vel, ang_vel]))
        self.world.step(render=False)
        self.ep_steps += 1

        observations, lidar_data, camera_data, mlp_obs = self.get_observations()

        if self.state_normalize == True:
            mlp_obs = self.de_normalize_states(mlp_obs)
        
        self.info = {}
        
        # camera_data = self.bot.rl_bot_camera.get_rgba()[:, :, :3]
        self.info['lidar_data'] = lidar_data
        self.info['camera_data'] = camera_data

        # mlp_obs = np.array(mlp_obs)
        reward, to_point_rew, reward_dict, ep_rew_dict = self.reward_manager.compute_reward(prev_bot_pos[:2], mlp_obs[0], mlp_obs[-1], lidar_data)
        done, done_reason = self.is_terminated(mlp_obs[0], mlp_obs[-1])
        if done:
            if done_reason == "timeout":
                reward = self.reward_manager.timeout_penalty
                to_point_rew = 0
            elif done_reason == "boundary_collision":
                reward += self.reward_manager.boundary_coll_penalty
                to_point_rew = 0
            elif done_reason == 'goal_reached':
                self.update_curriculum(done_reason == "goal_reached")
                print("Goal Reached!!!!")

        self.logger.info(f"Step: action={action}, reward={reward}, done={done}")
        wandb.log({
            'Action_lin_vel' : action[0],
            'Action_ang_vel' : action[1],
            "step_reward": reward,
            "to_point_reward": to_point_rew,
        })
        wandb.log(self.reward_manager.ep_reward_dict)

        return observations, reward, done, self.info

    def is_terminated(self, cur_bot_pos, goal_pos):
        if self.ep_steps >= self.max_episode_length:
            self.logger.info("Episode timed out")
            return True, "timeout"
        if self.reward_manager.check_goal_reached(cur_bot_pos, goal_pos):
            self.logger.info("Goal reached")
            return True, "goal_reached"
        if self.reward_manager.check_boundary(cur_bot_pos):
            self.logger.info("Boundary collision detected")
            return True, "boundary_collision"
        return False, None
    
    def update_curriculum(self, success):
        self.level_episodes[self.curriculum_level] += 1
        if success:
            self.level_success_rate[self.curriculum_level] += 1

        if self.level_episodes[self.curriculum_level] >= self.episodes_per_level:
            success_rate = self.level_success_rate[self.curriculum_level] / self.level_episodes[self.curriculum_level]
            if success_rate >= self.success_threshold and self.curriculum_level < 4:
                self.curriculum_level += 1
                self.reward_manager.update_curriculum_level(self.curriculum_level)
                print(f"Curriculum being updated to : {self.curriculum_level}")
                self.logger.info(f"Moving to curriculum level {self.curriculum_level}")
                wandb.log({"curriculum_level": self.curriculum_level})
            
            self.level_success_rate[self.curriculum_level] = 0
            self.level_episodes[self.curriculum_level] = 0

    # CURRICULUM GOAL POSE GENERATION
    def _gen_goal_pose(self):
        import random
        max_attempts = 100 
        for _ in range(max_attempts):
            new_pos = np.array([random.choice(list(set([x for x in np.linspace(-7.5, 7.6, 10000)]) - set(y for y in np.append(np.linspace(-2.6,-1.7,900), np.append(np.linspace(-0.8,0.4,1200), np.append(np.linspace(1.5,2.4,900), np.linspace(3.4,4.6,1200))))))),
                                random.choice(list(set([x for x in np.linspace(-5.5, 5.6, 14000)]) - set(y for y in np.append(np.linspace(-1.5,2.5,1000), np.linspace(-2.5,-5.6,3100))))),
                                0.0])
            
            if np.linalg.norm(new_pos[:2] - self.bot.rl_bot.start_pose[0][:2]) <= self.level_thresholds[self.curriculum_level]:
                return new_pos
        
        direction = (np.random.rand(2) - 0.5) * 2
        direction /= np.linalg.norm(direction)
        return np.array([
            self.bot.rl_bot.start_pose[0][0] + (direction[0] * self.level_thresholds[self.curriculum_level]),
            self.bot.rl_bot.start_pose[0][1] + (direction[1] * self.level_thresholds[self.curriculum_level]),
            0.0
        ])
    
    # STANDARD GOAL POSE GENERATION
    # def _gen_goal_pose(self):
    #     new_pos = np.array([random.choice(list(set([x for x in np.linspace(-7.5, 7.6, 10000)]) - set(y for y in np.append(np.linspace(-2.6,-1.7,900), np.append(np.linspace(-0.8,0.4,1200), np.append(np.linspace(1.5,2.4,900), np.linspace(3.4,4.6,1200))))))),
    #                         random.choice(list(set([x for x in np.linspace(-5.5, 5.6, 14000)]) - set(y for y in np.append(np.linspace(-1.5,2.5,1000), np.linspace(-2.5,-5.6,3100))))),
    #                         0.0])
    #     return new_pos
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]
    
    def render(self, mode='human'):
        self.world.render()

    def close(self):
        self.reward_manager.end_episode(self.episode_count) 
        self.logger.info("Environment closed")
        wandb.finish()
        self.kit.close()
    
    def quaternion_to_rotation_matrix(self, ori):
        w, x, y, z = ori
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
    
    
    # def get_lidar_image(self, lidar_data, mlp_obs, image_size=128):
    #     if lidar_data.size == 0:
    #         return None, None
        
    #     pos = mlp_obs[0]
    #     ori = mlp_obs[1]
    #     quat_matrix = self.quaternion_to_rotation_matrix(ori)
    #     rotated_data = np.dot(lidar_data[:, :2], quat_matrix[:2, :2].T)
    #     world_data = rotated_data + pos

    #     world_data[:, 0] = (world_data[:, 0] - self.world_min_max[0]) / (self.world_min_max[2] - self.world_min_max[0])
    #     world_data[:, 1] = (world_data[:, 1] - self.world_min_max[1]) / (self.world_min_max[3] - self.world_min_max[1])
    #     normalized_pos = (pos[:2] - self.world_min_max[:2]) / (self.world_min_max[2:] - self.world_min_max[:2])

    #     image = np.zeros((image_size, image_size), dtype=np.uint8)

    #     pixel_coords = (world_data * (image_size - 1)).astype(int)
    #     bot_pixel = (normalized_pos * (image_size - 1)).astype(int)

    #     valid_pixels = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < image_size) & \
    #                 (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < image_size)
    #     image[pixel_coords[valid_pixels, 1], pixel_coords[valid_pixels, 0]] = 128  # Gray dots for lidar points

    #     bot_size = 1
    #     bot_x, bot_y = bot_pixel
    #     image[max(0, bot_y-bot_size):min(image_size, bot_y+bot_size+1),
    #         max(0, bot_x-bot_size):min(image_size, bot_x+bot_size+1)] = 255  # White square for bot

    #     three_channel_image = np.stack([image] * 3, axis=0)
    #     return three_channel_image

# TEST PURPOSE CODE
# def run_episode(my_env):
#     lidar_dir = '/home/rah_m/new_lidar_data/new8/lidar_data'
#     camera_dir = '/home/rah_m/new_lidar_data/new8/camera_data'

#     for directory in [lidar_dir, camera_dir]:
#         if os.path.exists(directory):
#             shutil.rmtree(directory)
#         os.makedirs(directory)

#     for _ in range(20):
#         obs = my_env.reset()
#         done = False
#         total_rew = 0
#         i = 0
#         while i < my_env.max_episode_length:
#             if i % 1 == 0:
#                 actions = np.clip(np.random.uniform(-1, 1, 2), -1.0, 1.0)
#                 print(f"stepping at {i} with actions {actions}")
#                 obs, reward, done, info = my_env.step(actions)
#                 pos, ori = my_env.bot.rl_bot.get_world_pose()
#                 lidar_data = info['lidar_data']
#                 camera_data = info['camera_data']
#                 print(f"Lidar Data Shape: {lidar_data.shape}")
#                 print(f"Camera Data Shape: {camera_data.shape}")

#                 my_env.render()
#                 my_env.kit.update()

#                 np.save(f'{lidar_dir}/lidar_data_{i}_{pos}_{ori}.npy', lidar_data)
#                 np.save(f'{camera_dir}/camera_data_{i}_{pos}_{ori}.npy', camera_data)
#             my_env.kit.update()
#             if done:
#                 break
#             i += 1
#         # print(f"Episode Done with Reward: {my_env.reward_manager.get_total_reward()}")
#         break

def animated_loading():
    chars = ['.', '..', '...']
    for char in chars:
        sys.stdout.write('\r')
        sys.stdout.write("Social Navigation Training in Progress" + char)
        sys.stdout.flush()
        time.sleep(0.5)

class BestModelCallback(BaseCallback):
    def __init__(self, algo, save_path, verbose=0):
        super(BestModelCallback, self).__init__(verbose)
        self.algo = algo.upper()
        self.save_path = save_path
        self.best_mean_reward = -float('inf')

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            if self.model.ep_info_buffer and len(self.model.ep_info_buffer) > 0:
                rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
                mean_reward = np.mean(rewards)
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    path = os.path.join(self.save_path, f"best_SocNav_{self.algo}_ckpt.zip")
                    self.model.save(path)
                    
                    metadata = {
                        'mean_reward': float(mean_reward),
                        'timestamp': time.time()
                    }
                    metadata_path = os.path.join(self.save_path, f"best_SocNav_{self.algo}_ckpt_metadata.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f)
                    
                    print(f"Saving best model to {path} with mean reward {mean_reward}")
        return True

class IncrementalCheckpointCallback(CheckpointCallback):
    def __init__(self, algo, save_freq, save_path, name_prefix="rl_model", verbose=0):
        super().__init__(save_freq, save_path, name_prefix, verbose)
        self.algo = algo.upper()
        self.run_number = self.get_next_run_number(save_path)
        self.save_path = os.path.join(save_path, f"{self.algo}_{self.run_number}_ckpts")
        os.makedirs(self.save_path, exist_ok=True)

    def get_next_run_number(self, base_path):
        existing_runs = [d for d in os.listdir(base_path) if d.startswith(f"{self.algo}_") and d.endswith("_ckpts")]
        if not existing_runs:
            return 1
        run_numbers = [int(d.split("_")[1]) for d in existing_runs]
        return max(run_numbers) + 1
    
class WandbCallback(BaseCallback):
    def __init__(self, algo, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.step_count = 0
        self.algo = algo

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % 100 == 0:
            log_data = {
                "total_timesteps": self.num_timesteps,
                "learning_rate": self.model.learning_rate,
            }
            wandb.log(log_data)
            animated_loading()
        return True
    
class GradientAccumulationCallback(BaseCallback):
    def __init__(self, n_accumulate=4):
        super().__init__()
        self.n_accumulate = n_accumulate
        self.n_steps = 0

    def _on_step(self) -> bool:
        self.n_steps += 1
        if self.n_steps % self.n_accumulate == 0:
            if isinstance(self.model, PPO):
                self.model.policy.optimizer.step()
                self.model.policy.optimizer.zero_grad()
            elif isinstance(self.model, SAC):
                self.model.actor.optimizer.step()
                self.model.actor.optimizer.zero_grad()
                self.model.critic.optimizer.step()
                self.model.critic.optimizer.zero_grad()
                if self.model.ent_coef_optimizer is not None:
                    self.model.ent_coef_optimizer.step()
                    self.model.ent_coef_optimizer.zero_grad()
        return True

# class MixedPrecisionCallback(BaseCallback):
#     def __init__(self):
#         super().__init__()
#         self.scaler = amp.GradScaler()

#     def _on_step(self) -> bool:
#         if isinstance(self.model, PPO):
#             with amp.autocast():
#                 self.model._train_step(self.model.rollout_buffer.get(batch_size=self.model.batch_size))
#             self.scaler.step(self.model.policy.optimizer)
#             self.scaler.update()
#         elif isinstance(self.model, SAC):
#             with amp.autocast():
#                 self.model.train(gradient_steps=1, batch_size=self.model.batch_size)
#         return True
    
# class LearningRateScheduler(BaseCallback):
#     def __init__(self, initial_lr, min_lr=1e-6, decay_factor=0.99):
#         super().__init__()
#         self.initial_lr = initial_lr
#         self.min_lr = min_lr
#         self.decay_factor = decay_factor

#     def _on_step(self):
#         new_lr = max(self.initial_lr * (self.decay_factor ** (self.num_timesteps / 1000)), self.min_lr)
#         self.model.learning_rate = new_lr
#         return True

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256, algo="ppo", device="cuda", mlp_context=120, img_context=30):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=2) #Placeholder Feature dims for PyTorch call
        
        self.mlp_context = mlp_context
        self.img_context = img_context
        self.device = device

        extractors = { 'vector': {}, 'image': {} }
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if algo == "ppo":
                if key == "vector":
                    # MLP for our vector data
                    extractors[key]['core'] = nn.Sequential(
                        nn.Linear(13, 256),
                        nn.ReLU(),
                        nn.Linear(256, 512),
                        nn.ReLU(),
                        nn.Linear(512, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.ReLU()
                    )
                    extractors[key]['lstm'] = nn.LSTM(512, 128, num_layers=2)
                    total_concat_size += 128
                    
                elif key == "image":
                    # CNN for our image data (LiCam for now)
                    n_input_channels = 3
                    n_flatten_size = 1024
                    extractors[key]['core'] = nn.Sequential(
                        nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
                        nn.ReLU(),
                        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
                        nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same'),
                        nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
                        nn.ReLU(),
                        nn.Flatten(),
                        )
                    extractors[key]['lstm'] = nn.LSTM(n_flatten_size, 256, num_layers=2)
                    total_concat_size += cnn_output_dim

            elif algo == 'sac':
                if key == "vector":
                # MLP for our vector data
                    extractors[key]['core'] = nn.Sequential(
                        nn.Linear(13, 256),
                        nn.ReLU(),
                        nn.Linear(256, 512),
                        nn.ReLU(),
                        nn.Linear(512, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.ReLU()
                    )
                    extractors[key]['lstm'] = nn.LSTM(512, 128, num_layers=2)
                    total_concat_size += 128
                elif key == "image":
                    # CNN for our image data (LiCam for now)
                    n_input_channels = 3
                    n_flatten_size = 1024
                    extractors[key]['core'] = nn.Sequential(
                        nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
                        nn.ReLU(),
                        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
                        nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same'),
                        nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
                        nn.ReLU(),
                        nn.Flatten(),
                        )
                    extractors[key]['lstm'] = nn.LSTM(n_flatten_size, 256, num_layers=2)
                    total_concat_size += cnn_output_dim

        self.extractors = { 'vector': nn.ModuleDict(extractors['vector']).to(self.device), 'image': nn.ModuleDict(extractors['image']).to(self.device) }
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            if key == "vector":
                if observations[key].shape[0] == 1:
                    core_in = observations[key].squeeze(0).to(self.device)
                    core_out = extractor['core'](core_in)
                    lstm_out = extractor['lstm'](core_out)[-1][0]
                    encoded_tensor_list.append(lstm_out[-1].unsqueeze(0))
                else:
                    core_in = observations[key].view(-1, 13).to(self.device)
                    core_out = extractor['core'](core_in).view(self.mlp_context, -1, 512)
                    lstm_out = extractor['lstm'](core_out)[1][0]

                    encoded_tensor_list.append(lstm_out[-1])
            else:
                if observations[key].shape[0] == 1:
                    core_in = observations[key].squeeze(0).to(self.device)
                    core_out = extractor['core'](core_in)
                    lstm_out = extractor['lstm'](core_out)[-1][0]
                    encoded_tensor_list.append(lstm_out[-1].unsqueeze(0))

                    # print(f"MPL_Encoded Tensor List Length: {len(encoded_tensor_list)}")
                    # print(f"MPL_Encoded Tensor List Shape: {encoded_tensor_list[0].shape}")
                    # print(f"MPL_Encoded Tensor List Shape: {encoded_tensor_list[1].shape}")

                else:
                    core_in = observations[key].view(-1, 3, 64, 64).to(self.device)
                    batch_size = core_in.shape[0]
                    split_size = batch_size // 32
                    core_in_n = th.split(core_in, split_size)

                    core_out_n = []
                    for i in range(split_size):
                        core_out_n.append(extractor['core'](core_in_n[i]).view(self.img_context, -1, 1024))

                    # Combine the results
                    core_out = th.cat(core_out_n, dim=1)

                    # Process through LSTM
                    lstm_out = extractor['lstm'](core_out)[-1][0]
                    encoded_tensor_list.append(lstm_out[-1])

                    # core_in = observations[key].view(-1, 3, 64, 64).to(self.device)                    
                    # core_out = extractor['core'](core_in).view(self.img_context, -1, 1024)
                    # lstm_out = extractor['lstm'](core_out)[-1][0]
                    # encoded_tensor_list.append(lstm_out[-1])

        return th.cat(encoded_tensor_list, dim=1) # encoded tensor is the batch dimension

# def setup_logging():
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     return logging.getLogger(__name__)

def get_checkpoint(algo: str, ckpt_dir: str, best: bool = True):
    pattern = os.path.join(ckpt_dir, f"{algo.upper()}_*_ckpts", f"{'best_' if best else ''}SocNav_{algo.upper()}_ckpt*.zip")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    if best:
        best_checkpoint = None
        best_reward = -float('inf')
        for checkpoint in checkpoints:
            metadata_path = checkpoint.replace('.zip', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if metadata['mean_reward'] > best_reward:
                        best_reward = metadata['mean_reward']
                        best_checkpoint = checkpoint
        return best_checkpoint
    else:
        return max(checkpoints, key=os.path.getctime)

def create_model(algo: str, my_env, policy_kwargs: dict, tensor_log_dir: str):
    if algo.lower() == 'ppo':
        return PPO(
                PPOMultiPolicy,
                my_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                n_steps=2048,
                batch_size=256,
                learning_rate=5e-4,
                gamma=0.99,
                ent_coef=0.1,
                clip_range=0.3,
                n_epochs=10,
                device="cuda",
                gae_lambda=1.0,
                max_grad_norm=0.9,
                vf_coef=0.95,
                use_sde=False,
                tensorboard_log=tensor_log_dir
            )
    elif algo.lower() == 'sac':
        return SAC(
                SACMultiPolicy,
                my_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                buffer_size=5000,
                learning_rate=1e-4,
                gamma=0.99,
                batch_size=512,
                tau=0.005,
                ent_coef='auto_0.1',
                target_update_interval=1,
                train_freq=(1, 'episode'),
                gradient_steps=-1,
                learning_starts=5000,
                use_sde=False,
                sde_sample_freq=-1,
                use_sde_at_warmup=False,
                optimize_memory_usage=False,
                device="cuda",
                tensorboard_log=tensor_log_dir,
            )
    
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

def main():
    parser = argparse.ArgumentParser(description="Train Omni Isaac SocNav Agent")
    parser.add_argument('--algo', type=str, choices=['ppo', 'sac'], default='ppo', help='RL algorithm to use')
    parser.add_argument('--botname', type=str, choices=['jackal', 'carter', 'nova_carter'], default='jackal', help='Choose the bot to train')
    parser.add_argument('--ckpt', type=str, choices=['latest', 'best'], default='best', help='Choose the checkpoint to resume from')
    parser.add_argument('--mlp_context', type=int, default=30, help='Length of the MLP context')
    parser.add_argument('--img_context', type=int, default=10, help='Length of the image Context')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--state_normalize', action='store_true', help='Normalize Observation States')
    parser.add_argument('--resume_checkpoint', action='store_true', help='Resume from the best checkpoint')
    args = parser.parse_args()

    try:
        with wandb.init(project="Mod_Isaac_Soc_Nav", name=f"{args.algo.upper()}_training") as run:
            log_dir = "/home/rah_m/Mod_SocNav_Logs"
            ckpt_log_dir = os.path.join(log_dir, "Checkpoints")
            tensor_log_dir = os.path.join(log_dir, "Tensorboard")
            os.makedirs(ckpt_log_dir, exist_ok=True)

            # print(f"Headless args : {args.headless}")
            # print(f"Algo args : {args.algo}")

            my_env = SocEnv(algo=args.algo, botname=args.botname, headless=args.headless, state_normalize=args.state_normalize, mlp_context=args.mlp_context, img_context=args.img_context, logdir=log_dir)

            best_checkpoint = get_checkpoint(args.algo, ckpt_log_dir, best=True)
            latest_checkpoint = get_checkpoint(args.algo, ckpt_log_dir, best=False)
            if args.ckpt == 'best':
                checkpoint_to_load = best_checkpoint
            else:
                checkpoint_to_load = latest_checkpoint

            # checkpoint_to_load = best_checkpoint or latest_checkpoint

            policy_kwargs = {
                'ppo' : dict(net_arch=[dict(pi=[256, 512, 512, 256], vf=[256, 512, 512, 256])], activation_fn=th.nn.Tanh,
                             features_extractor_class=CustomCombinedExtractor,
                            features_extractor_kwargs=dict(cnn_output_dim=256, algo="ppo", device="cuda", mlp_context=args.mlp_context, img_context=args.img_context)),
                'sac' : dict(net_arch=dict(pi=[256, 512, 512, 256], qf=[256, 512, 512, 256]), activation_fn=th.nn.ReLU, use_sde=False, log_std_init=-3,
                             features_extractor_class=CustomCombinedExtractor,
                             features_extractor_kwargs=dict(cnn_output_dim=256, algo="sac", device='cuda', mlp_context=args.mlp_context, img_context=args.img_context))}[args.algo.lower()]

            model = create_model(args.algo, my_env, policy_kwargs, tensor_log_dir)

            if args.resume_checkpoint:
                print(f"Checkpoint to load: {checkpoint_to_load}")
                # logger.info(f"Loading checkpoint: {checkpoint_to_load}")
                model = model.load(checkpoint_to_load, env=my_env)
            else:
                print("No Checkpoint Loading!!!")

            checkpoint_callback = IncrementalCheckpointCallback(algo=args.algo, save_freq=2000, save_path=ckpt_log_dir, 
                                                                name_prefix=f"SocNav_{args.algo.upper()}_ckpt")
            wandb_callback = WandbCallback(algo=args.algo)
            best_model_callback = BestModelCallback(algo=args.algo, save_path=os.path.join(ckpt_log_dir, f"{args.algo.upper()}_{checkpoint_callback.run_number}_ckpts"))
            gradient_accumulation_callback = GradientAccumulationCallback(n_accumulate=4)
            # mixed_precision_callback = MixedPrecisionCallback()
            model.learn(total_timesteps=500000000, callback=[checkpoint_callback, wandb_callback, best_model_callback, gradient_accumulation_callback])

            final_model_path = os.path.join(ckpt_log_dir, f"{args.algo.upper()}_{checkpoint_callback.run_number}_final", f"Soc_Nav_{args.algo.upper()}_Policy")
            model.save(final_model_path)
            # logger.info(f"Training completed! Final model saved to {final_model_path}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        # logger.exception(f"An error occurred during training: {e}")
    finally:
        wandb.finish()
        my_env.close()

# if __name__ == '__main__':
if __name__ == '__main__':
    main()








# BASIC RL PIPELINE CODE

#     parser = argparse.ArgumentParser(description="Let's Train our Omni Isaac SocNav Agent!")
#     parser.add_argument('--algo', type=str, choices=['ppo', 'sac'], default='ppo', help='RL algorithm to use (ppo or sac)')
#     args = parser.parse_args()

#     wandb.init(project="social_navigation", name=f"{args.algo.upper()}_training")    

#     log_dir = "/home/rah_m/SocNav_Logs"
#     ckpt_log_dir = "/home/rah_m/SocNav_Logs/Checkpoints"
#     tensor_log_dir = "/home/rah_m/SocNav_Logs/Tensorboard"

#     os.makedirs(ckpt_log_dir, exist_ok=True)

#     my_env = SocEnv(headless=True, logdir=log_dir)

#     best_checkpoint = get_best_checkpoint(args.algo, ckpt_log_dir)
#     latest_checkpoint = get_latest_checkpoint(args.algo, ckpt_log_dir)
#     checkpoint_to_load = best_checkpoint or latest_checkpoint

#     # input_dim = my_env.observation_space.shape[0]
#     if args.algo.lower() == 'ppo':
#         policy_kwargs = dict(
#             activation_fn=th.nn.Tanh,
#             net_arch=[
#                 dict(pi=[128, 128, 128], vf=[128, 128, 128])
#             ]
#         )
#     elif args.algo.lower() == 'sac':
#         policy_kwargs = dict(
#             net_arch=dict(pi=[256, 256, 256], qf=[256, 256, 256]),
#             activation_fn=th.nn.ReLU,
#             use_sde=False,
#             log_std_init=-3,
#         )

#     total_timesteps = 50000000

#     with wandb.init(project="social_navigation", name=f"{args.algo.upper()}_training") as run:
#         if args.algo.lower() == 'ppo':
#             model = PPO(
#                 PPOMlpPolicy,
#                 my_env,
#                 policy_kwargs=policy_kwargs,
#                 verbose=1,
#                 n_steps=2560,
#                 batch_size=64,
#                 learning_rate=0.000125,
#                 gamma=0.97,
#                 ent_coef=7.5e-08,
#                 clip_range=0.3,
#                 n_epochs=5,
#                 device="cuda",
#                 gae_lambda=1.0,
#                 max_grad_norm=0.9,
#                 vf_coef=0.95,
#                 tensorboard_log=tensor_log_dir
#             )
#         elif args.algo.lower() == 'sac':
#             model = SAC(
#                 SACMlpPolicy,
#                 my_env,
#                 policy_kwargs=policy_kwargs,
#                 verbose=1,
#                 buffer_size=1000000,
#                 learning_rate=3e-4,
#                 gamma=0.99,
#                 batch_size=256,
#                 tau=0.005,
#                 ent_coef='auto',
#                 target_update_interval=1,
#                 train_freq=1,
#                 gradient_steps=1,
#                 learning_starts=10000,
#                 use_sde=False,
#                 sde_sample_freq=-1,
#                 use_sde_at_warmup=False,
#                 tensorboard_log=tensor_log_dir,
#                 device="cuda"
#             )

#         if checkpoint_to_load:
#             print(f"Loading checkpoint: {checkpoint_to_load}")
#             model = model.load(checkpoint_to_load, env=my_env)

#     checkpoint_callback = IncrementalCheckpointCallback(algo=args.algo, save_freq=5000, save_path=ckpt_log_dir, 
#                                                         name_prefix=f"SocNav_{args.algo.upper()}_ckpt")
#     wandb_callback = WandbCallback(algo=args.algo)
#     best_model_callback = BestModelCallback(algo=args.algo, save_path=os.path.join(ckpt_log_dir, f"{args.algo.upper()}_{checkpoint_callback.run_number}_ckpts"))

#     model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, wandb_callback, best_model_callback])

#     # Save the final model
#     final_model_path = os.path.join(ckpt_log_dir, f"{args.algo.upper()}_{checkpoint_callback.run_number}_final", f"Soc_Nav_{args.algo.upper()}_Policy")
#     model.save(final_model_path)
#     print(f"\nTraining completed! Final model saved to {final_model_path}")

#     wandb.finish()
#     my_env.close()