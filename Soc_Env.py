import argparse
import sys
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import math
import asyncio
import threading
import carb

class SocEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(
        self,
        skip_frame=10,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=10000000,
        seed=0,
        headless=True,
    ) -> None:
        
        from isaacsim import SimulationApp

        CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}

        parser = argparse.ArgumentParser("Usd Load sample")
        parser.add_argument(
            "--usd_path", type=str, help="Path to usd file, relative to our default assets folder", required=True
        )
        parser.add_argument("--headless", default=False, action="store_true", help="Run stage headless")
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
        import asyncio
        import threading

        from pxr import Usd, Vt, Gf

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
        # from RL_Bot import RLBot
        from RLBL import RLBot
        # from Pegasus_App import PegasusApp
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
        # lidar_config = 'RPLIDAR_S2E'
    
        self.bot = RLBot(self.world, self.kit, self.timeline, self.assets_root_path)
        self.controller = RLBotController()
        self.act = RLBotAct(self.bot.rl_bot, self.controller, n_steps=5)
        # print("BOT INSITIALIZED")
        # self.bot.run()
        print("Bot Initialised!")
        inav = omni.anim.navigation.core.acquire_interface()
        print("Navmesh Created!")
        self.people = PegasusApp(self.world, self.kit, self.timeline)
        print("People initialised!")
        self.kit.update()

        # ASYNCIO INITIALIZATION
        self.loop = asyncio.get_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()

        # BOT PARAMETERS
        self.bot.rl_bot.start_pose = None
        self.bot.rl_bot.goal_pose = None
        self.max_velocity = 1.5
        self.max_angular_velocity = np.pi

        # RL PARAMETERS
        self.seed(seed)
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=float("inf"), high=float("inf"), shape=(16,), dtype=np.float32)

        #print(f"Action Space : {self.action_space}")
        #print(f"Observation Space : {self.observation_space}")

        self.reset_counter = 0

        # REWARD PARAMETERS
        self.total_reward = {"ep_reward": 0, "ep_social_reward": 0, "ep_path_reward": 0}
        self.ep_steps = 0
        self.total_steps = 0
        self.max_episode_length = max_episode_length

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _run_coroutine(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def reset(self):
        self.kit.update()
        self.bot.bot_reset()
        self.world.reset(True)
        self.bot.rl_bot.start_pose = self.bot.rl_bot.get_world_pose()
        self.bot.rl_bot.goal_pose = self._gen_goal_pose()
        self.kit.update()
        # self.people.reset() # To be implemented If needed
        #print(f"Bot's New Pose : {self.bot.rl_bot.get_world_pose()}")
        #print(f"Bot's Goal Pose : {self.bot.rl_bot.goal_pose}")

        observations = self.get_observations()

        self.total_reward = {"ep_reward": 0, "ep_social_reward": 0, "ep_path_reward": 0}
        self.ep_steps = 0

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

    # async def step(self, action):  # action = [forward, angular] both within [-1, 1]
    #     prev_bot_pos, _ = self.bot.rl_bot.get_world_pose()
    #     print("Running LiDAR Instance!")
    #     depth, azimuth, intensity, linear_depth = await self.bot.get_lidar_param()
        
    #     print("Lidar Data Shapes:")
    #     print(f"Depth: {depth.shape}")
    #     print(f"Azimuth: {azimuth.shape}")
    #     print(f"Intensity: {intensity.shape}")
    #     print(f"Linear Depth: {linear_depth.shape}")

    #     lin_vel = action[0] * self.max_velocity
    #     ang_vel = action[1] * self.max_angular_velocity

    #     for i in range(self._skip_frame):
    #         self.act.move_bot(vals=np.array([lin_vel, ang_vel]))
    #         self.world.step(render=False)

    #     observations = self.get_observations()
    #     info = {
    #         'lidar_depth': depth,
    #         'lidar_azimuth': azimuth,
    #         'lidar_intensity': intensity,
    #         'lidar_linear_depth': linear_depth
    #     }
    #     done = False
    #     if self.world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
    #         done = True
    #     goal_pos = self.bot.rl_bot.goal_pose
    #     cur_bot_pos, _ = self.bot.rl_bot.get_world_pose()
    #     prev_dist_to_goal = np.linalg.norm(goal_pos - prev_bot_pos)
    #     cur_dist_to_goal = np.linalg.norm(goal_pos - cur_bot_pos)
    #     reward = prev_dist_to_goal - cur_dist_to_goal
    #     if cur_dist_to_goal < 0.1:
    #         done = True
    #     return observations, reward, done, info
    
    def step(self, action):
        prev_bot_pos, _ = self.bot.rl_bot.get_world_pose()
        print("Running LiDAR Instance!")
        # depth, azimuth, intensity, linear_depth = self._run_coroutine(self.bot.get_lidar_param())
        depth, azimuth, intensity, linear_depth = self.bot.get_lidar_param()
        # depth, azimuth, intensity, linear_depth = self.loop.run_until_complete(self.bot.get_lidar_param())
        print("Lidar Data Shapes:")
        print(f"Depth: {depth.shape}")
        print(f"Azimuth: {azimuth.shape}")
        print(f"Intensity: {intensity.shape}")
        print(f"Linear Depth: {linear_depth.shape}")

        lin_vel = action[0] * self.max_velocity
        ang_vel = action[1] * self.max_angular_velocity

        for i in range(self._skip_frame):
            self.act.move_bot(vals=np.array([lin_vel, ang_vel]))
            self.world.step(render=False)

        observations = self.get_observations()
        info = {
            'lidar_depth': depth,
            'lidar_azimuth': azimuth,
            'lidar_intensity': intensity,
            'lidar_linear_depth': linear_depth
        }
        done = False
        if self.world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True
        goal_pos = self.bot.rl_bot.goal_pose
        cur_bot_pos, _ = self.bot.rl_bot.get_world_pose()
        prev_dist_to_goal = np.linalg.norm(goal_pos - prev_bot_pos)
        cur_dist_to_goal = np.linalg.norm(goal_pos - cur_bot_pos)
        reward = prev_dist_to_goal - cur_dist_to_goal
        if cur_dist_to_goal < 0.1:
            done = True
        return observations, reward, done, info                  

    # def step(self, action):  # action = [forward, angular] both within [-1, 1]
    #     # self.world.step(True)
    #     prev_bot_pos, _ = self.bot.rl_bot.get_world_pose()
    #     print("Running LiDAR Instance!")
    #     lidar_data = self.loop.run_until_complete(self.bot.run_lidar_instance())
    #     # lidar_data = self.bot.run_lidar_instance()
    #     print(lidar_data.shape)
    #     print(len(lidar_data))

    #     lin_vel = action[0] * self.max_velocity
    #     ang_vel = action[1] * self.max_angular_velocity

    #     for i in range(self._skip_frame):
    #         self.act.move_bot(vals=np.array([lin_vel, ang_vel]))
    #         self.world.step(render=False)

    #     observations = self.get_observations()
    #     info = {}
    #     done = False
    #     if self.world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
    #         done = True
    #     goal_pos = self.bot.rl_bot.goal_pose
    #     cur_bot_pos, _ = self.bot.rl_bot.get_world_pose()
    #     prev_dist_to_goal = np.linalg.norm(goal_pos - prev_bot_pos)
    #     cur_dist_to_goal = np.linalg.norm(goal_pos - cur_bot_pos)
    #     reward = prev_dist_to_goal - cur_dist_to_goal
    #     if cur_dist_to_goal < 0.1:
    #         done = True
    #     return observations, reward, done, info
    #     # All actiohns and reward computations

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
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.kit.close()

# def test_task(env):
#     i = 1
#     reset_needed = False
#     while env.kit.is_running():
#         env.timeline.play()
#         env.world.step(render=True)
#         if env.world.is_stopped() and not reset_needed:
#             reset_needed = True
#         if env.world.is_playing():
#             if i % 500 != 0:
#                 with open('/home/rah_m/Isaac_World_Files/lidar_data.txt', 'a+') as f:
#                     f.write(str(env.bot.rl_bot_lidar.get_current_frame()))
#                     f.write('\n')
#                 if i % 150 == 0:
#                     rand_val = np.random.uniform(-2,2,1)
#                     rand_val = np.append(rand_val, np.random.uniform(-np.pi, np.pi, 1))
#                     #print(f"Applied Action Values : {rand_val}")
#                     env.act.move_bot(vals=rand_val)
#                     with open('/home/rah_m/Isaac_World_Files/camera_data.txt', 'a+') as f:
#                         f.write(str(env.bot.rl_bot_camera.get_current_frame()))
#                         f.write('\n')
#                     imgplot = plt.imshow(env.bot.rl_bot_camera.get_rgba()[:, :, :3])
#                     plt.show()
#             else:
#                 # carb.log_warn("PegasusApp Simulation App is closing.")
#                 #print("Simulation Done!")
#                 # env.timeline.stop()
#                 # env.world.stop()
#                 env.reset()
#                 # env.kit.update()
#                 # env.world.reset(True)
#                 # env.kit.update()
#                 # env.kit.close()
#                 # break
#             i += 1
#     # env.world.stop()
#     #print('loop is done!')

def run_episode(my_env):
    for _ in range(20):
        obs = my_env.reset()
        done = False
        total_rew = 0
        i = 0
        while not done:
            if i % 10 == 0:
                actions = np.random.uniform(-1, 1, 2)
                actions = np.clip(actions, -1.5, 1.5)
                obs, reward, done, info = my_env.step(actions)
                total_rew += reward
                my_env.render()
                my_env.kit.update()
                # Access lidar data if needed
                lidar_depth = info['lidar_depth']
                with open(f'/home/rah_m/Isaac_Lidar/lidar_depth_{i}.txt', 'w') as f:
                    f.write(str(lidar_depth))
                    f.write('\n')
                print(lidar_depth.shape )
                lidar_azimuth = info['lidar_azimuth']
                with open(f'/home/rah_m/Isaac_Lidar/lidar_azimuth_{i}.txt', 'w') as f:
                    f.write(str(lidar_azimuth))
                    f.write('\n')
                print(lidar_azimuth.shape)
                lidar_intensity = info['lidar_intensity']
                with open(f'/home/rah_m/Isaac_Lidar/lidar_intensity_{i}.txt', 'w') as f:
                    f.write(str(lidar_intensity))
                    f.write('\n')
                print(lidar_intensity.shape)
                lidar_linear_depth = info['lidar_linear_depth']
                with open(f'/home/rah_m/Isaac_Lidar/lidar_linear_depth_{i}.txt', 'w') as f:
                    f.write(str(lidar_linear_depth))
                    f.write('\n')
                print(lidar_linear_depth.shape)
            i += 1

if __name__ == '__main__':

    my_env = SocEnv()
    print("Environment Created!")
    my_env.reset()
    print("Environment Reset!")

    asyncio.run(run_episode(my_env=my_env))
    my_env.close()

    # for _ in range(20):
    #     obs = my_env.reset()
    #     done = False
    #     total_rew = 0
    #     i = 0
    #     while not done:
    #         if i % 10000 == 0:
    #             #print(f"Step : {i}")
    #             actions = np.random.uniform(-1, 1, 2)
    #             #print(f"Actions : {actions}")
    #             obs, reward, done, info = my_env.step(actions)
    #             #print(f"Observations : {obs}")
    #             #print(f"Reward : {reward}")
    #             total_rew += reward
    #             #print(f"Total Reward : {total_rew}")
    #             my_env.render()
    #             my_env.kit.update()
    #         i += 1

    # my_env.close()