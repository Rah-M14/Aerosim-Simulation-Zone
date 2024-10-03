
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.sensor import Camera
from omni.isaac.sensor import LidarRtx
import omni.kit.commands
import omni.replicator.core as rep

import asyncio
from pxr import Gf
import numpy as np
import random

# Nova Carter In our Case
class RLBot():
    def __init__(self, simulation_app, world, timeline, assets_root_path):

        self.kit = simulation_app
        self.world = world
        self.timeline = timeline
        self.assets_root_path = assets_root_path

        self.rl_bot_asset_path = self.assets_root_path + "/Isaac/Robots/Carter/nova_carter_sensors.usd"
        self.rl_bot_carter = WheeledRobot(    
            prim_path="/World/Nova_Carter",
            name="RL_Bot",
            wheel_dof_names=["joint_caster_left", "joint_caster_right", "joint_wheel_left", "joint_wheel_right"],
            wheel_dof_indices=[3, 4, 5, 6],
            create_robot=True,
            usd_path=self.rl_bot_asset_path,
            position=np.array([0.4, -0.4,0]))
        self.world.scene.add(self.rl_bot_carter) 

        self.rl_bot = ArticulationView(prim_paths_expr="/World/Nova_Carter", name="RL_Bot")
        self.rl_bot = self.world.scene.get_object("RL_Bot")
        print(f"RL_Bot : {self.rl_bot}")
        print("RL Bot in the World")

        # CARTER CAMERA
        self.rl_bot_camera = Camera(prim_path="/World/Nova_Carter/chassis_link/front_owl/camera",
                                    name='Carter_Camera',
                                    frequency=30,
                                    resolution=(512,512))
        self.rl_bot_camera.initialize()
        self.kit.update()
        self.rl_bot_camera.initialize()
        print(f"RL_Bot Camera : {self.rl_bot_camera}")
        self.world.initialize_physics()

        # CARTER LiDAR
        # self.rl_bot_lidar = self.world.scene.add(
        #     LidarRtx(prim_path="/World/Nova_Carter/chassis_link/front_RPLidar/RPLIDAR_S2E", 
        #             name="Carter_Lidar"))
        # print(f"RL_Bot LiDAR : {self.rl_bot_lidar}")
        # print("performing reset")
        # self.world.reset()
        # print("reset done")
        # self.rl_bot_lidar.add_range_data_to_frame()
        # self.rl_bot_lidar.add_point_cloud_data_to_frame()
        # self.rl_bot_lidar.enable_visualization()
        # print("RL Bot Initialized!")
        # # Bot Parameters

        # lidar_config = "RPLIDAR_S2E_Soc_Nav"
        lidar_config = "Soc_Lidar"

        # 1. Create The Camera
        _, self.rl_bot_lidar = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path="/RPLIDAR_S2E",
            parent="/World/Nova_Carter/chassis_link/XT_32/RPLIDAR_Req",
            config=lidar_config,
            translation=(0, 0, 0.0),
            orientation=Gf.Quatd(1,0,0,0),
        )
        render_product = rep.create.render_product(self.rl_bot_lidar.GetPath(), [1, 1])

        self.rl_bot_lidar.annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
        self.rl_bot_lidar.annotator.attach(render_product)

        writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
        writer.attach(render_product)

        # self.rl_bot.current_state = self.rl_bot.get_world_pose()

    # async def lidar_data(self):
    #     await rep.orchestrator.step_async()
    #     self.rl_bot_lidar.data = self.rl_bot_lidar.annotator.get_data()
    #     # print(f"Lidar Data: {self.rl_bot_lidar.data}")
    #     print(f"Lidar Data : {self.rl_bot_lidar.data['data']}")
    #     print(f"Lidar Data Type: {type(self.rl_bot_lidar.data['data'])}")
    #     print(f"Lidar Data Shape: {self.rl_bot_lidar.data['data'].shape}")

    def lidar_data(self):
        self.kit.update()
        self.timeline.pause()
        rep.orchestrator.step()
        self.rl_bot_lidar.data = self.rl_bot_lidar.annotator.get_data()
        self.timeline.play()
        self.kit.update()
        return self.rl_bot_lidar.data['data']

    def bot_reset(self):
        valid_pos_x = random.choice(list(set([x for x in np.linspace(-7.5, 7.6, 10000)]) - set(y for y in np.append(np.linspace(-2.6,-1.7,900), np.append(np.linspace(-0.8,0.4,1200), np.append(np.linspace(1.5,2.4,900), np.linspace(3.4,4.6,1200)))))))
        valid_pos_y = random.choice(list(set([x for x in np.linspace(-5.5, 5.6, 14000)]) - set(y for y in np.append(np.linspace(-1.5,2.5,1000), np.linspace(-2.5,-5.6,3100)))))
        new_pos = np.array([valid_pos_x, valid_pos_y, 0.0])

        self.rl_bot.set_default_state(position=new_pos, orientation=np.array([1, 0, 0, 0]))
        print("Bot is reset!")
    
    # def bot_current_pose(self):
    #     return self.rl_bot.get_world_pose()

    