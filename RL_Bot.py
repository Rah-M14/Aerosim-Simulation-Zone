
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.sensor import Camera
from omni.isaac.sensor import LidarRtx

# from RL_Bot_Control import RLBotController

import numpy as np

# Nova Carter In our Case
class RLBot():
    def __init__(self, simulation_app, world, assets_root_path):

        self.kit = simulation_app
        self.world = world
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
        self.rl_bot_lidar = self.world.scene.add(
            LidarRtx(prim_path="/World/Nova_Carter/chassis_link/front_RPLidar/RPLIDAR_S2E", 
                    name="Carter_Lidar"))
        print(f"RL_Bot LiDAR : {self.rl_bot_lidar}")
        print("performing reset")
        self.world.reset()
        print("reset done")
        self.rl_bot_lidar.add_range_data_to_frame()
        self.rl_bot_lidar.add_point_cloud_data_to_frame()
        self.rl_bot_lidar.enable_visualization()

        # CARTER CONTROLLER
        # self.rl_bot_controller = RLBotController()
        # print("Controller Created!")

        print("RL Bot Initialized!")