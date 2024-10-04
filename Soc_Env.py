import argparse
import sys
import os
import time
import shutil
from datetime import datetime
import carb
import logging
import wandb

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch as th

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

# from Soc_Env import SocEnv

class SocEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(
        self,
        logdir,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=1000,
        seed=0,
        headless=True,
    ) -> None:
        
        from isaacsim import SimulationApp

        CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": True, "renderer": "RayTracedLighting"}

        parser = argparse.ArgumentParser("Usd Load sample")
        parser.add_argument(
            "--usd_path", type=str, help="Path to usd file, relative to our default assets folder", required=True
        )
        parser.add_argument("--headless", default=True, action="store_true", help="Run stage headless")
        parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")

        args, unknown = parser.parse_known_args()
        CONFIG["headless"] = args.headless
        self.kit = SimulationApp(launch_config=CONFIG)
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        
        import omni
        import matplotlib.pyplot as plt
        # from gymnasium import spaces
        import torch
        import random

        from pxr import Usd, UsdPhysics, UsdGeom, Vt, Gf, Sdf

        import asyncio
        from stable_baselines3 import PPO

        from omni.isaac.core import World, SimulationContext
        from omni.isaac.core.scenes import Scene

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

        from omni.isaac.core.tasks.base_task import BaseTask

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

        from RL_Bot_Control import RLBotController, RLBotAct
        from RL_Bot import RLBot
        from SocRewards import SocialReward
        from Mod_Pegasus_App import PegasusApp

        self.assets_root_path = get_assets_root_path()
        if self.assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            self.kit.close()
            sys.exit()
        usd_path = self.assets_root_path + args.usd_path

        try:
            result = is_file(usd_path)
        except:
            result = False

        if result:
            omni.usd.get_context().open_stage(usd_path)
            self.stage = omni.usd.get_context().get_stage()
            self.scene = UsdPhysics.Scene.Define(self.stage, "/physicsScene")
            # stage = Usd.Stage.Open('omniverse://localhost/Projects/SIMS/PEOPLE_SIMS/New_Core.usd')
        else:
            carb.log_error(
                f"the usd path {usd_path} could not be opened, please make sure that {args.usd_path} is a valid usd file in {self.assets_root_path}"
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
    
        self.bot = RLBot(simulation_app=self.kit, world=self.world, timeline=self.timeline, assets_root_path=self.assets_root_path)
        self.controller = RLBotController()
        self.act = RLBotAct(self.bot.rl_bot, self.controller, n_steps=5)
        print("Bot Initialised!")
        inav = omni.anim.navigation.core.acquire_interface()
        print("Navmesh Created!")
        self.people = PegasusApp(self.world, self.stage, self.kit, self.timeline)
        print("People initialised!")
        self.kit.update()

        # BOT PARAMETERS
        self.bot.rl_bot.start_pose = None
        self.bot.rl_bot.goal_pose = None
        self.max_velocity = 1.0
        self.max_angular_velocity = np.pi

        # RL PARAMETERS
        self.seed(seed)
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=float("inf"), high=float("inf"), shape=(16,), dtype=np.float32)

        self.ep_steps = 0
        self.episode_count = 0
        self.info = {}

        self.logdir = logdir
        self.setup_logging()

        # REWARD PARAMETERS
        self.reward_manager = SocialReward(self.logdir)
        self.max_episode_length = max_episode_length

        wandb.init(project="SocNav_Omni", name="environment_logging")

    def setup_logging(self):
        log_dir = "home/rah_m/Soc_Train_Logs/Env_Logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"soc_env_{timestamp}.log")
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def reset(self):
        print("For some reason reset is being called")
        self.episode_count += 1
        self.reward_manager.end_episode(self.episode_count)
        self.kit.update()
        self.bot.bot_reset()
        self.world.reset(True)
        self.bot.rl_bot.start_pose = self.bot.rl_bot.get_world_pose()
        self.bot.rl_bot.goal_pose = self._gen_goal_pose()
        self.kit.update()

        observations = self.get_observations()
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
        return np.concatenate([bot_world_pos,
                                bot_world_ori,
                                bot_lin_vel,
                                bot_ang_vel,
                                goal_world_pos])

    def step(self, action):
        prev_bot_pos, _ = self.bot.rl_bot.get_world_pose()
        
        lin_vel = action[0] * self.max_velocity
        ang_vel = action[1] * self.max_angular_velocity
        for _ in range(self._skip_frame):
            self.act.move_bot(vals=np.array([lin_vel, ang_vel]))
            self.world.step(render=False)
            self.ep_steps += 1

        observations = self.get_observations()
        self.info = {}
        
        lidar_data = self.bot.get_denoised_lidar_data()
        if lidar_data.size == 0:
            print("Warning: No valid LiDAR data available")
            lidar_data = None

        camera_data = self.bot.rl_bot_camera.get_rgba()[:, :, :3]
        self.info['lidar_data'] = lidar_data
        self.info['camera_data'] = camera_data

        goal_pos = self.bot.rl_bot.goal_pose
        cur_bot_pos, _ = self.bot.rl_bot.get_world_pose()

        done, done_reason = self.is_terminated(cur_bot_pos, goal_pos)
        if done == True and done_reason == "timeout":
            reward = self.reward_manager.timeout_penalty
            to_point_rew = 0
        elif done == True and done_reason == "goal_reached":
            reward = self.reward_manager.goal_reward
            to_point_rew = 0
        elif done == True and done_reason == "boundary_collision":
            reward = self.reward_manager.boundary_coll_penalty
            to_point_rew = 0
        else:
            reward, to_point_rew, reward_dict, ep_rew_dict = self.reward_manager.compute_reward(prev_bot_pos, cur_bot_pos, goal_pos, lidar_data)

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
        if np.linalg.norm(goal_pos - cur_bot_pos) < self.reward_manager.goal_dist_thresold:
            self.logger.info("Goal reached")
            return True, "goal_reached"
        if self.reward_manager.check_boundary_collision(cur_bot_pos):
            self.logger.info("Boundary collision detected")
            return True, "boundary_collision"
        return False, None

    def _gen_goal_pose(self):
        import random
        new_pos = np.array([random.choice(list(set([x for x in np.linspace(-7.5, 7.6, 10000)]) - set(y for y in np.append(np.linspace(-2.6,-1.7,900), np.append(np.linspace(-0.8,0.4,1200), np.append(np.linspace(1.5,2.4,900), np.linspace(3.4,4.6,1200))))))),
                            random.choice(list(set([x for x in np.linspace(-5.5, 5.6, 14000)]) - set(y for y in np.append(np.linspace(-1.5,2.5,1000), np.linspace(-2.5,-5.6,3100))))),
                            0.0])
        return new_pos
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]
    
    def render(self, mode='human'):
        self.world.render()

    def close(self):
        self.reward_manager.end_episode() 
        self.logger.info("Environment closed")
        wandb.finish()
        self.kit.close()

# TEST PURPOSE CODE
# def run_episode(my_env):
#     lidar_dir = '/home/rah_m/new_lidar_data/new6/lidar_data'
#     camera_dir = '/home/rah_m/new_lidar_data/new6/camera_data'

#     for directory in [lidar_dir, camera_dir]:
#         if os.path.exists(directory):
#             shutil.rmtree(directory)
#         os.makedirs(directory)

#     for _ in range(20):
#         obs = my_env.reset()
#         done = False
#         total_rew = 0
#         i = 0
#         while i < my_env.max_ep_steps:
#             if i % 1 == 0:
#                 actions = np.clip(np.random.uniform(-1, 1, 2), -1.0, 1.0)
#                 print(f"stepping at {i} with actions {actions}")
#                 obs, reward, done, info = my_env.step(actions)
#                 pos, _ = my_env.bot.rl_bot.get_world_pose()
#                 lidar_data = info['lidar_data']
#                 camera_data = info['camera_data']
#                 print(f"Lidar Data Shape: {lidar_data.shape}")
#                 print(f"Camera Data Shape: {camera_data.shape}")

#                 my_env.render()
#                 my_env.kit.update()

#                 np.save(f'{lidar_dir}/lidar_data_{i}_{pos}.npy', lidar_data)
#                 np.save(f'{camera_dir}/camera_data_{i}_{pos}.npy', camera_data)
#             my_env.kit.update()
#             if done:
#                 break
#             i += 1
#         print(f"Episode Done with Reward: {my_env.reward_manager.get_total_reward()}")
#         break

def animated_loading():
    chars = ['.', '..', '...']
    for char in chars:
        sys.stdout.write('\r')
        sys.stdout.write("Social Navigation Training in Progress" + char)
        sys.stdout.flush()
        time.sleep(0.5)

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % 1000 == 0:
            wandb.log({
                "total_timesteps": self.num_timesteps,
                "learning_rate": self.model.learning_rate,
            })
            animated_loading()
        return True

class IncrementalCheckpointCallback(CheckpointCallback):
    def __init__(self, save_freq, save_path, name_prefix="rl_model", verbose=0):
        super().__init__(save_freq, save_path, name_prefix, verbose)
        self.run_number = self.get_next_run_number(save_path)
        self.save_path = os.path.join(save_path, f"PPO_{self.run_number}_ckpts")
        os.makedirs(self.save_path, exist_ok=True)

    def get_next_run_number(self, base_path):
        existing_runs = [d for d in os.listdir(base_path) if d.startswith("PPO_") and d.endswith("_ckpts")]
        if not existing_runs:
            return 1
        run_numbers = [int(d.split("_")[1]) for d in existing_runs]
        return max(run_numbers) + 1
    
if __name__ == '__main__':

    wandb.init(project="social_navigation", name="training")    
    log_dir = "/home/rah_m/SocNav_Logs/Checkpoints"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    my_env = SocEnv(headless=True, logdir=log_dir)

    input_dim = my_env.observation_space.shape[0]
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=[
            dict(pi=[128, 128, 128], vf=[128, 128, 128])
        ]
    )

    policy = MlpPolicy
    total_timesteps = 50000000

    with wandb.init(project="social_navigation", name="training") as run:
        model = PPO(
            policy,
            my_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=2560,
            batch_size=64,
            learning_rate=0.000125,
            gamma=0.97,
            ent_coef=7.5e-08,
            clip_range=0.3,
            n_epochs=5,
            device="cuda",
            gae_lambda=1.0,
            max_grad_norm=0.9,
            vf_coef=0.95,
            tensorboard_log=log_dir
        )

        checkpoint_callback = IncrementalCheckpointCallback(save_freq=5000, save_path=log_dir, name_prefix="SocNav_ckpt")
        wandb_callback = WandbCallback()
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, wandb_callback])
    
    print("\nTraining completed!")
    model.save(os.path.join(log_dir, f"PPO_{checkpoint_callback.run_number}_final", "Soc_Nav_Policy"))
    wandb.finish()
    my_env.close()