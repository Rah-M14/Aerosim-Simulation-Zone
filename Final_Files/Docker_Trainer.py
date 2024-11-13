import argparse
import glob
import json
import logging
import os
import pickle
import shutil
import sys
import time
from datetime import datetime

import carb
import torch as th
import wandb
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo import MultiInputPolicy as PPOMultiPolicy
from stable_baselines3.sac import MultiInputPolicy as SACMultiPolicy

th.cuda.empty_cache()
th.backends.cudnn.benchmark = True

import gym
import numpy as np
from gym import spaces


class SocEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        algo,
        botname,
        state_normalize,
        gpus,
        logdir,
        mlp_context,
        img_context,
        headless,
        skip_frame=1,
        physics_dt=1/1,
        rendering_dt=1/1,
        max_episode_length=200,
        seed=0,
    ) -> None:

        from isaacsim import SimulationApp

        CONFIG = {
            "width": 1280,
            "height": 720,
            "sync_loads": True,
            "headless": headless,
            "active_gpu": gpus[0],
            "physics_gpu": gpus[0],
            "multi_gpu": False,
            "renderer": "RayTracedLighting",
        }

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

        import math
        import random
        import time
        from collections import deque

        import matplotlib.pyplot as plt
        import omni
        import omni.isaac.core.utils.numpy.rotations as rot_utils

        # from omni.isaac.core.tasks.base_task import BaseTask
        import omni.kit.viewport.utility

        # import omni.anim.navigation.core as nav
        import omni.usd

        # from gymnasium import spaces
        import torch as th
        import torch.nn as nn
        from omni.isaac.core import SimulationContext, World

        # from omni.isaac.core.scenes import Scene
        # from omni.isaac.core.utils import stage
        from omni.isaac.core.utils.extensions import enable_extension
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.rotations import quat_to_euler_angles

        # from omni.isaac.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
        # from omni.isaac.core.utils.stage import add_reference_to_stage
        # from omni.isaac.core.utils.prims import create_prim
        # from omni.isaac.core.utils.viewports import set_camera_view
        from omni.isaac.core.utils.stage import is_stage_loading
        from omni.isaac.nucleus import get_assets_root_path, is_file
        from stable_baselines3 import PPO
        from ultralytics import YOLO

        # import omni.replicator.core as rep

        EXTENSIONS_PEOPLE = [
            "omni.anim.people",
            "omni.anim.navigation.bundle",
            "omni.anim.navigation.core",
            "omni.anim.timeline",
            "omni.anim.graph.bundle",
            "omni.anim.graph.core",
            "omni.anim.graph.ui",
            "omni.anim.retarget.bundle",
            "omni.anim.retarget.core",
            "omni.anim.retarget.ui",
            "omni.kit.scripting",
            "omni.graph.io",
            "omni.anim.curve.core",
        ]

        for ext_people in EXTENSIONS_PEOPLE:
            enable_extension(ext_people)

        # import omni.anim.navigation.core as nav
        self.kit.update()

        from LiDAR_Feed import get_licam_image
        from Mod_Pegasus_App import PegasusApp
        from New_RL_Bot import RLBot
        from New_RL_Bot_Control import (
            CustomWheelBasePoseController,
            RLBotAct,
            RLBotController,
        )

        from Adv_SocRewards_Coll import SocialReward

        self.assets_root_path = get_assets_root_path()
        if self.assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            self.kit.close()
            sys.exit()

        # current_world_usd = "omniverse://localhost/Projects/SIMS/PEOPLE_SIMS/New_Core.usd"
        # usd_path = self.assets_root_path + current_world_usd

        # usd_path = "/home/rahm/.local/share/ov/pkg/isaac-sim-4.2.0/standalone_examples/api/omni.isaac.kit/Final_WR_World/New_Core.usd"
        usd_path = "/isaac-sim/standalone_examples/api/omni.isaac.kit/final_wr_world/new_core"

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
                f"the usd path {usd_path} could not be opened, please make sure that {usd_path} is a valid usd file in {self.assets_root_path}"
            )
            self.kit.close()
            sys.exit()
        self.kit.update()
        self.kit.update()

        # print("Loading stage...")

        while is_stage_loading():
            self.kit.update()
        # print("Loading Complete")

        self.headless = headless
        self.world = World(
            physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0
        )
        self.sim_context = SimulationContext()
        self.timeline = omni.timeline.get_timeline_interface()
        self.world.initialize_physics()
        print(f"the current physics dt is : {self.world.get_physics_dt()}")
        print(f"the current rendering dt is : {self.world.get_rendering_dt()}")
        # self.world.set_physics_step_size(step_size=self._dt * 60) 
        # self.world.set_gpu_dynamics_enabled(enabled=True)
        # print("Waiting Room with Physics initialization in the World!")

        enable_extension("omni.isaac.debug_draw")

        self.botname = botname
        self.bot = RLBot(
            simulation_app=self.kit,
            world=self.world,
            timeline=self.timeline,
            assets_root_path=self.assets_root_path,
            botname=self.botname,
        )
        # self.core_controller = RLBotController(botname=self.botname)
        # self.controller = CustomWheelBasePoseController(
        #     name="cool_controller",
        #     open_loop_wheel_controller=self.core_controller,
        #     is_holonomic=False,
        # )

        # self.act = RLBotAct(self.kit, self.bot.rl_bot, self.controller)
        # self.world.add_physics_callback(
        #     "sending_actions", callback_fn=self.send_robot_actions
        # )
        print("Bot Initialised!")
        inav = omni.anim.navigation.core.acquire_interface()
        print("Navmesh Created!")
        self.people = PegasusApp(self.world, self.stage, self.kit, self.timeline)
        print("People initialised!")
        self.kit.update()

        # WORLD PARAMETERS

        # self.next_pos ,self.next_ori = None, None
        # self.human_motion = 0

        self.world_limits = np.array([20.0, 14.0])
        self.world_min_max = np.array([-10, -7, 10, 7])  # (min_x, min_y, max_x, max_y)

        # self.carter_vel_limits = np.array([7.0, 7.0])
        # self.carter_ang_vel_limits = np.array([np.pi, np.pi])
        self.state_normalize = state_normalize
        
        self.timestep = 0
        self.total_timesteps = 0
        self.frame_num = 0

        # BOT PARAMETERS
        # self.prev_pos = self.people.person_list[0]._state.position
        # self.cur_pos = self.people.person_list[0]._state.position
        # self.bot_vel = 0
        # self.prev_bots_pos, _ = self.bot.rl_bot.get_world_pose()

        self.bot_length = 0.508 #Length in metres
        self.bot_l = self.bot_length*2
        self.bot_theta = np.pi/9
        
        self.bot.rl_bot.start_pose = None
        self.bot.rl_bot.goal_pose = None
        
        # self.max_velocity = 1.0
        # self.max_angular_velocity = np.pi * 4
        # self.act_steps = 5
        # self.pos_tol = 0.05  # Thiss tolerance should be greater than the position_tol in the controller

        self.yolo_model = YOLO("yolo11n.pt")

        # RL PARAMETERS
        self.seed(seed)

        mlp_zeros = np.concatenate([np.zeros(2), np.zeros(4), np.zeros(2), np.zeros(1)])
        self.mlp_context_frame_length = mlp_context
        self.img_context_frame_length = img_context
        self.mlp_context_frame = deque(
            [mlp_zeros for _ in range(self.mlp_context_frame_length)],
            maxlen=self.mlp_context_frame_length,
        )
        self.img_context_frame = deque(
            [np.zeros((3, 64, 64)) for _ in range(self.img_context_frame_length)],
            maxlen=self.img_context_frame_length,
        )

        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1]),
            high=np.array([1, 1]),
            shape=(2,),
            dtype=np.float32,
        )  # (l, theta)
        self.observation_space = spaces.Dict(
            {
                "vector": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.mlp_context_frame_length, 9),
                    dtype=np.float32,
                ),
                "image": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.img_context_frame_length, 3, 64, 64),
                    dtype=np.float32,
                ),
            }
        )

        self.ep_steps = 0
        self.episode_count = 0
        self.info = {}

        self.algo = algo

        self.logdir = os.path.join(logdir, self.algo.upper())
        self.env_log_dir = os.path.join(self.logdir, "Env_Logs")
        self.logger = self.setup_logging()
        print(f"Logger initialized: {self.logger}")
        self.logger.info("This is a test log message from __init__")

        # REWARD PARAMETERS
        self.reward_manager = SocialReward(self.logdir)
        self.max_episode_length = max_episode_length

        # CURRICULUM APPROACH PARAMETERS5, -np.pi
        self.curriculum_level = 4
        self.level_thresholds = {1: 3.0, 2: 7.0, 3: 11.0, 4: float("inf")}
        self.level_success_rate = {1: 0, 2: 0, 3: 0, 4: 0}
        self.level_episodes = {1: 0, 2: 0, 3: 0, 4: 0}
        self.success_threshold = 0.85
        self.episodes_per_level = 100
        self.reward_manager.update_curriculum_level(self.curriculum_level)

    def setup_logging(self):
        try:
            os.makedirs(self.env_log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.env_log_dir, f"social_env_{timestamp}.log")

            with open(log_file, "w") as f:
                f.write("Initializing log file\n")

            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)

            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
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
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            logger.warning(f"Falling back to console logging due to error: {e}")
            return logger

    def reset(self):
        self.episode_count += 1
        self.reward_manager.end_episode(self.episode_count)
        self.kit.update()
        self.bot.bot_reset()
        self.world.reset(True)
        self.timestep = np.array([0.0])

        self.bot.rl_bot.start_pose = self.bot.rl_bot.get_world_pose()
        if not self.headless:
            print(f"Start pose : {self.bot.rl_bot.start_pose}")
        self.bot.rl_bot.goal_pose = self._gen_goal_pose()
        if not self.headless:
            print(f"Goal Pose: {self.bot.rl_bot.goal_pose}")

        observations, _, _, _ = self.get_observations()
        self.ep_steps = 0

        self.logger.info(f"Episode {self.episode_count} started")
        wandb.log({"episode": self.episode_count})
        return observations

    def get_observations(self):
        bot_world_pos, bot_world_ori = self.bot.rl_bot.get_world_pose()
        goal_world_pos = self.bot.rl_bot.goal_pose
        timestep = self.timestep
        mlp_states_un = [bot_world_pos[:2], bot_world_ori, goal_world_pos[:2], timestep]
        if self.state_normalize == True:
            mlp_states = [
                bot_world_pos[:2] / self.world_limits,
                bot_world_ori / np.linalg.norm(bot_world_ori),
                goal_world_pos[:2] / self.world_limits,
                timestep / self.max_episode_length,
            ]

        self.mlp_context_frame.append(np.concatenate(mlp_states))
        mlp_combined_context = self.get_mlp_combined_context()

        lidar_data = self.bot.get_denoised_lidar_data()
        camera_data = self.bot.rl_bot_camera.get_rgba()[:, :, :3]
        if lidar_data.size == 0:
            lidar_data = None
            licam_image = np.zeros((3, 64, 64))
            img_combined_context = th.zeros((self.img_context_frame_length, 3, 64, 64))
        else:
            li_cam_image = self.get_li_cam_image(
                lidar_data,
                camera_data,
                mlp_states_un,
                self.yolo_model,
                self.world_min_max,
                project_camera=True,
                image_size=64,
            )
            if self.timestep % 2 == 0:
                self.img_context_frame.append(li_cam_image)
            img_combined_context = self.get_img_combined_context()

        return (
            {"vector": mlp_combined_context, "image": img_combined_context},
            lidar_data,
            camera_data,
            mlp_states,
        )

    def get_mlp_combined_context(self):
        return th.tensor(np.array(self.mlp_context_frame, dtype=np.float32))

    def get_img_combined_context(self):
        return th.tensor(np.array(self.img_context_frame, dtype=np.float32))

    def get_li_cam_image(
        self,
        lidar_data,
        camera_image,
        mlp_obs,
        yolo_model,
        world_min_max,
        project_camera=False,
        image_size=64,
    ):
        from LiDAR_Feed import get_licam_image

        return get_licam_image(
            lidar_data,
            camera_image,
            mlp_obs,
            yolo_model,
            world_min_max,
            project_camera,
            image_size,
        )

    def de_normalize_states(self, observation):
        return [
            observation[0] * self.world_limits,
            observation[1],
            observation[2] * self.world_limits,
            observation[3] * self.max_episode_length,
        ]

    def next_coords(self, actions, position, orientation):
        from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat

        l, theta = actions[0]*self.bot_l, actions[1]*self.bot_theta

        _, _, current_yaw = quat_to_euler_angles(orientation)
        world_theta = current_yaw + theta
        
        x1, y1 = l*np.cos(world_theta), l*np.sin(world_theta)
        next_pos = np.array([x1, y1, 0.0])

        orien = euler_angles_to_quat(np.array([0.0, 0.0, world_theta]))
        
        return (position + next_pos, orien)

    def step(
        self, action
    ):  # action = [l, theta], l in [-0.5, 1] and theta in [-1, 1]
        wandb.log({"Simulation_timesteps" : self.sim_context.current_time})
        try:
            prev_bot_pos, prev_bot_ori = self.bot.rl_bot.get_world_pose()
        except Exception as e:
            print(f"Warning in step: Failed to get initial robot pose: {e}")
            return (
                self.get_observations()[0],
                self.reward_manager.timeout_penalty,
                True,
                {},
            )
        
        prev_bot_pos, prev_bot_ori = self.bot.rl_bot.get_world_pose()

        self.timestep += 1.0
        self.total_timesteps += 1

        self.next_pos, self.next_ori = self.next_coords(action, prev_bot_pos, prev_bot_ori)
        self.bot.rl_bot.set_world_pose(position=self.next_pos, orientation=self.next_ori)

        self.ep_steps += 1

        observations, lidar_data, camera_data, mlp_obs = self.get_observations()
        if self.state_normalize == True:
            mlp_obs = self.de_normalize_states(mlp_obs)

        self.info = {}
        self.info["lidar_data"] = lidar_data
        self.info["camera_data"] = camera_data

        reward, to_point_rew, reward_dict, ep_rew_dict = (
            self.reward_manager.compute_reward(
                prev_bot_pos[:2], mlp_obs[0], mlp_obs[-2], lidar_data
            )
        )
        done, done_reason = self.is_terminated(mlp_obs[0], mlp_obs[-1])
        if done:
            if done_reason == "timeout":
                reward = self.reward_manager.timeout_penalty
                wandb.log({"timeout": self.reward_manager.timeout_penalty})
                to_point_rew = 0
            elif done_reason == "boundary_collision":
                reward += self.reward_manager.boundary_coll_penalty
                wandb.log(
                    {"boundary_collision": self.reward_manager.boundary_coll_penalty}
                )
                to_point_rew = 0
            elif done_reason == "goal_reached":
                self.update_curriculum(done_reason == "goal_reached")
                print("Goal Reached!!!!")

        self.logger.info(f"Step: action={action}, reward={reward}, done={done}")
        wandb.log(
            {
                "L": action[0],
                "Theta": action[1],
                "step_reward": reward,
                "to_point_reward": to_point_rew,
                "step_total_timesteps": self.total_timesteps,
            }
        )
        wandb.log(self.reward_manager.ep_reward_dict)

        return observations, reward, done, self.info

    def is_terminated(self, cur_bot_pos, goal_pos):
        if self.ep_steps >= self.max_episode_length:
            self.logger.info("Episode timed out")
            wandb.log({"timeout": 0})
            return True, "timeout"
        if self.reward_manager.check_goal_reached(cur_bot_pos, goal_pos):
            self.logger.info("Goal reached")
            wandb.log({"boundary_collision": 0})
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
            success_rate = (
                self.level_success_rate[self.curriculum_level]
                / self.level_episodes[self.curriculum_level]
            )
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
            new_pos = np.array(
                [
                    random.choice(
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
                    random.choice(
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
                    ),
                    0.0,
                ]
            )

            if (
                np.linalg.norm(new_pos[:2] - self.bot.rl_bot.start_pose[0][:2])
                <= self.level_thresholds[self.curriculum_level]
            ):
                return new_pos

        direction = (np.random.rand(2) - 0.5) * 2
        direction /= np.linalg.norm(direction)
        return np.array(
            [
                self.bot.rl_bot.start_pose[0][0]
                + (direction[0] * self.level_thresholds[self.curriculum_level]),
                self.bot.rl_bot.start_pose[0][1]
                + (direction[1] * self.level_thresholds[self.curriculum_level]),
                0.0,
            ]
        )

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def render(self, mode="human"):
        self.world.render()

    def close(self):
        self.reward_manager.end_episode(self.episode_count)
        self.logger.info("Environment closed")
        wandb.finish()
        self.kit.close()

    def quaternion_to_rotation_matrix(self, ori):
        w, x, y, z = ori
        return np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ]
        )


def animated_loading():
    chars = [".", "..", "..."]
    for char in chars:
        sys.stdout.write("\r")
        sys.stdout.write("Social Navigation Training in Progress" + char)
        sys.stdout.flush()
        time.sleep(0.5)


class BestModelCallback(BaseCallback):
    def __init__(self, algo, save_path, save_replay_buffer=True, freq=5000, verbose=0):
        super(BestModelCallback, self).__init__(verbose)
        self.algo = algo.upper()
        self.save_path = save_path
        self.best_mean_reward = -float("inf")
        self.save_replay_buffer = save_replay_buffer
        self.freq = freq

    def _on_step(self) -> bool:
        rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
        mean_reward = np.mean(rewards)
        if self.n_calls % 1000 == 0:
            if self.model.ep_info_buffer and len(self.model.ep_info_buffer) > 0:
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    path = os.path.join(
                        self.save_path, f"best_SocNav_{self.algo}_ckpt.zip"
                    )
                    self.model.save(path)

                    if (
                        self.save_replay_buffer
                        and hasattr(self.model, "replay_buffer")
                        and self.model.replay_buffer is not None
                        and self.algo != "SAC"
                    ):
                        replay_buffer_path = os.path.join(
                            self.save_path, f"best_SocNav_{self.algo}_Replay_Buffer.pkl"
                        )
                        self.model.save_replay_buffer(replay_buffer_path)
                        if self.verbose > 1:
                            print(
                                f"Saving model replay buffer checkpoint to {replay_buffer_path}"
                            )

                    metadata = {
                        "mean_reward": float(mean_reward),
                        "timestamp": time.time(),
                    }
                    metadata_path = os.path.join(
                        self.save_path, f"best_SocNav_{self.algo}_ckpt_metadata.json"
                    )
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f)

                    print(f"Saving best model to {path} with mean reward {mean_reward}")

        if self.n_calls % self.freq == 0:
            gen_path = os.path.join(
                self.save_path, f"SocNav_{self.algo}_{self.n_calls}_ckpt.zip"
            )
            self.model.save(gen_path)

            replay_buffer_path = os.path.join(
                self.save_path, f"Latest_SocNav_{self.algo}_Replay_Buffer.pkl"
            )
            self.model.save_replay_buffer(replay_buffer_path)
            if self.verbose > 1:
                print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            metadata = {"mean_reward": float(mean_reward), "timestamp": time.time()}
            metadata_path = os.path.join(
                self.save_path, f"SocNav_{self.algo}_{self.n_calls}_ckpt_metadata.json"
            )
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)
            print(f"Saving model to {gen_path}")
        return True


class IncrementalCheckpointCallback(CheckpointCallback):
    def __init__(self, algo, save_freq, save_path, name_prefix="rl_model", verbose=0):
        super().__init__(save_freq, save_path, name_prefix, verbose)
        self.algo = algo.upper()
        self.run_number = self.get_next_run_number(save_path)
        self.save_path = os.path.join(save_path, f"{self.algo}_{self.run_number}_ckpts")
        os.makedirs(self.save_path, exist_ok=True)

    def get_next_run_number(self, base_path):
        existing_runs = [
            d
            for d in os.listdir(base_path)
            if d.startswith(f"{self.algo}_") and d.endswith("_ckpts")
        ]
        if not existing_runs:
            return 1
        run_numbers = [int(d.split("_")[1]) for d in existing_runs]
        return max(run_numbers) + 1


# class WandbCallback(BaseCallback):
#     def __init__(self, algo, verbose=0):
#         super(WandbCallback, self).__init__(verbose)
#         self.step_count = 0
#         self.algo = algo

#     def _on_step(self) -> bool:
#         self.step_count += 1
#         if self.step_count % 20 == 0:
#             # log_data = {
#             #     "callback_total_timesteps": self.step_count,
#             #     "learning_rate": self.model.learning_rate,
#             # }
#             # wandb.log(log_data)
#             # animated_loading()
#         return True


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


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_output_dim: int = 256,
        algo="ppo",
        mlp_context=32,
        img_context=16,
    ):
        super(CustomCombinedExtractor, self).__init__(
            observation_space, features_dim=2
        )  # Placeholder Feature dims for PyTorch call
        import torch.nn as nn

        self._observation_space = observation_space
        self.mlp_context = mlp_context
        self.img_context = img_context
        self.img_channels = 3
        self.vec_lstm_out_size = 32
        self.img_lstm_out_size = 128
        self.n_flatten_size = 256

        # self.device = device

        self.vec_lstm = nn.LSTM(9, self.vec_lstm_out_size, num_layers=1)
        self.img_cnn = nn.Sequential(
            nn.Conv2d(self.img_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.img_lstm = nn.LSTM(
            self.n_flatten_size, self.img_lstm_out_size, num_layers=2
        )

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if algo == "ppo":
                if key == "vector":
                    total_concat_size += self.vec_lstm_out_size
                elif key == "image":
                    total_concat_size += self.img_lstm_out_size

            elif algo == "sac":
                if key == "vector":
                    total_concat_size += self.vec_lstm_out_size
                elif key == "image":
                    total_concat_size += self.img_lstm_out_size

        self._features_dim = total_concat_size

    def get_device(self, network):
        return next(network.parameters()).device

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        model_device = self.get_device(self.img_lstm)

        for key, subspace in self._observation_space.spaces.items():
            if key == "vector":
                if observations[key].shape[0] == 1:
                    core_in = observations[key].squeeze(0)
                    # print("vec device nb", core_in.device)
                    lstm_out = self.vec_lstm(core_in)[-1][0]
                    encoded_tensor_list.append(lstm_out[-1].unsqueeze(0))
                else:
                    core_in = (
                        observations[key].view(self.mlp_context, -1, 9).to(model_device)
                    )
                    # print("vec device batch", core_in.device)
                    lstm_out = self.vec_lstm(core_in)[1][0]
                    encoded_tensor_list.append(lstm_out[-1])

            else:
                if observations[key].shape[0] == 1:
                    core_in = observations[key].squeeze(0)
                    # print("img device nb", core_in.device)
                    core_out = self.img_cnn(core_in)
                    lstm_out = self.img_lstm(core_out)[-1][0]
                    encoded_tensor_list.append(lstm_out[-1].unsqueeze(0))
                else:
                    core_in = observations[key].to(model_device)
                    # print("img device batch", core_in.device)
                    batch_size = core_in.shape[0]
                    mini_batch_size = 32
                    num_mini_batches = batch_size // mini_batch_size
                    core_in = core_in.view(
                        num_mini_batches, mini_batch_size, -1, 3, 64, 64
                    )

                    core_out_n = []
                    for i in range(num_mini_batches):
                        cnn_out = self.img_cnn(core_in[i].view(-1, 3, 64, 64))
                        core_out_n.append(
                            cnn_out.view(
                                mini_batch_size, self.img_context, 256
                            ).swapaxes(0, 1)
                        )

                    core_out = th.cat(core_out_n, dim=1)

                    lstm_out = self.img_lstm(core_out)[-1][0]
                    encoded_tensor_list.append(lstm_out[-1])

        return th.cat(
            encoded_tensor_list, dim=1
        )  # encoded tensor is the batch dimension


def get_checkpoint(algo: str, ckpt_dir: str, best: bool = True):
    pattern = os.path.join(
        ckpt_dir,
        f"{algo.upper()}_*_ckpts",
        f"{'best_' if best else ''}SocNav_{algo.upper()}_ckpt*.zip",
    )
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None

    if best:
        best_checkpoint = None
        best_reward = -float("inf")
        for checkpoint in checkpoints:
            metadata_path = checkpoint.replace(".zip", "_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    if metadata["mean_reward"] > best_reward:
                        best_reward = metadata["mean_reward"]
                        best_checkpoint = checkpoint
        return best_checkpoint
    else:
        return max(checkpoints, key=os.path.getctime)


def load_model_and_replay_buffer(
    algo, model, my_env, ckpt_path, replay_buffer_path=None
):
    model.load(ckpt_path, env=my_env)
    if replay_buffer_path and os.path.exists(replay_buffer_path):
        with open(replay_buffer_path, "rb") as f:
            model.replay_buffer = pickle.load(f)
        print(f"Replay buffer loaded from {replay_buffer_path}")

    return model


def create_model(algo: str, my_env, gpus, policy_kwargs: dict, tensor_log_dir: str):
    if algo.lower() == "ppo":
        return PPO(
            PPOMultiPolicy,
            my_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            ent_coef=0.1,
            clip_range=0.3,
            n_epochs=20,
            device=f"cuda:{gpus[0]}",
            gae_lambda=1.0,
            max_grad_norm=0.9,
            vf_coef=0.95,
            use_sde=False,
            tensorboard_log=tensor_log_dir,
        )
    elif algo.lower() == "sac":
        return SAC(
            SACMultiPolicy,
            my_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            buffer_size=10000,
            learning_rate=3e-4,
            gamma=0.99,
            batch_size=512,
            tau=0.005,
            ent_coef="auto_0.1",
            target_update_interval=1,
            train_freq=(1, "episode"),
            gradient_steps=-1,
            learning_starts=1,
            use_sde=False,
            sde_sample_freq=-1,
            use_sde_at_warmup=False,
            optimize_memory_usage=False,
            device=f"cuda:{gpus[0]}",
            tensorboard_log=tensor_log_dir,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")


def main():
    parser = argparse.ArgumentParser(description="Train Omni Isaac SocNav Agent")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "sac"],
        default="ppo",
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--botname",
        type=str,
        choices=["jackal", "carter", "nova_carter"],
        default="jackal",
        help="Choose the bot to train",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        choices=["latest", "best"],
        default="latest",
        help="Choose the checkpoint to resume from",
    )
    parser.add_argument(
        "--storage_path",
        type=str,
        default="/data/SocNav_Logs",
        help="Choose the path to store teh logs and ckpts",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        nargs="+",
        type=int,
        help="Specofy the [Active Gpu/Model Gpu, Physics GPU]",
        required=True,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Choose the checkpoint to resume from",
    )
    parser.add_argument(
        "--mlp_context", type=int, default=32, help="Length of the MLP context"
    )
    parser.add_argument(
        "--img_context", type=int, default=16, help="Length of the image Context"
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--state_normalize", action="store_true", help="Normalize Observation States"
    )
    parser.add_argument(
        "--resume_checkpoint",
        action="store_true",
        help="Resume from the best checkpoint",
    )
    args = parser.parse_args()

    try:
        with wandb.init(
            project="Mod_Isaac_Soc_Nav", name=f"{args.algo.upper()}_training"
        ) as run:
            gpus = args.gpu
            log_dir = args.storage_path
            ckpt_log_dir = os.path.join(log_dir, "Checkpoints")
            tensor_log_dir = os.path.join(log_dir, "Tensorboard")
            os.makedirs(ckpt_log_dir, exist_ok=True)

            device = f"cuda:{gpus[0]}"
            th.device(device)

            my_env = SocEnv(
                algo=args.algo,
                botname=args.botname,
                headless=args.headless,
                state_normalize=args.state_normalize,
                gpus=gpus,
                mlp_context=args.mlp_context,
                img_context=args.img_context,
                logdir=log_dir,
            )

            print(f"Checkpoint Given to be Loaded : {args.ckpt_path}")

            best_checkpoint = get_checkpoint(args.algo, ckpt_log_dir, best=True)
            latest_checkpoint = get_checkpoint(args.algo, ckpt_log_dir, best=False)
            if args.ckpt_path:
                checkpoint_to_load = args.ckpt_path
            elif args.ckpt == "best":
                checkpoint_to_load = best_checkpoint
            else:
                checkpoint_to_load = latest_checkpoint

            policy_kwargs = {
                "ppo": dict(
                    net_arch=[dict(pi=[256, 512, 512, 256], vf=[256, 512, 512, 256])],
                    activation_fn=th.nn.Tanh,
                    features_extractor_class=CustomCombinedExtractor,
                    features_extractor_kwargs=dict(
                        cnn_output_dim=256,
                        algo="ppo",
                        mlp_context=args.mlp_context,
                        img_context=args.img_context,
                    ),
                ),
                "sac": dict(
                    net_arch=dict(pi=[256, 512, 512, 256], qf=[256, 512, 512, 256]),
                    activation_fn=th.nn.ReLU,
                    use_sde=False,
                    log_std_init=-3,
                    features_extractor_class=CustomCombinedExtractor,
                    features_extractor_kwargs=dict(
                        cnn_output_dim=256,
                        algo="sac",
                        mlp_context=args.mlp_context,
                        img_context=args.img_context,
                    ),
                ),
            }[args.algo.lower()]

            model = create_model(args.algo, my_env, gpus, policy_kwargs, tensor_log_dir)

            if args.resume_checkpoint:
                print(f"Checkpoint to load: {checkpoint_to_load}")
                rep_buf_path = os.path.join(
                    f"{os.sep}".join(checkpoint_to_load.split("/")[:-1]),
                    f"best_SocNav_{args.algo.upper()}_Replay_Buffer.pkl",
                )
                model = load_model_and_replay_buffer(
                    algo=args.algo,
                    model=model,
                    my_env=my_env,
                    ckpt_path=checkpoint_to_load,
                    replay_buffer_path=rep_buf_path,
                )
            else:
                print("No Checkpoint Loading!!!")

            checkpoint_callback = IncrementalCheckpointCallback(
                algo=args.algo,
                save_freq=200,
                save_path=ckpt_log_dir,
                name_prefix=f"SocNav_{args.algo.upper()}_ckpt",
            )
            # wandb_callback = WandbCallback(algo=args.algo)
            best_model_callback = BestModelCallback(
                algo=args.algo,
                save_path=os.path.join(
                    ckpt_log_dir,
                    f"{args.algo.upper()}_{checkpoint_callback.run_number}_ckpts",
                ),
            )
            gradient_accumulation_callback = GradientAccumulationCallback(
                n_accumulate=4
            )
            eval_callback = EvalCallback(
                my_env,
                best_model_save_path=os.path.join(ckpt_log_dir, "Eval_Best_Model"),
                log_path=os.path.join(ckpt_log_dir, "Eval_Logs"),
                eval_freq=5000,
                n_eval_episodes=5,
                deterministic=True,
                render=False,
            )

            model.learn(
                total_timesteps=500000000,
                progress_bar=True,
                callback=[
                    checkpoint_callback,
                    best_model_callback,
                    eval_callback,
                    # wandb_callback,
                    gradient_accumulation_callback,
                ],
            )

            final_model_path = os.path.join(
                ckpt_log_dir,
                f"{args.algo.upper()}_{checkpoint_callback.run_number}_final",
                f"Soc_Nav_{args.algo.upper()}_Policy",
            )
            model.save(final_model_path)

    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        wandb.finish()
        my_env.close()

if __name__ == "__main__":
    main()