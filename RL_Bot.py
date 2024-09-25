import omni.kit.viewport.utility
from omni.isaac.core import World, SimulationContext
from omni.isaac.core.utils import stage
from omni.isaac.core.scenes import Scene
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers import DifferentialController, WheelBasePoseController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.controllers import BaseController
from omni.isaac.sensor import Camera
from omni.isaac.sensor import LidarRtx
# from omni.isaac.range_sensor import _range_sensor 
from omni.isaac.range_sensor._range_sensor import acquire_lidar_sensor_interface
from omni.isaac.nucleus import get_assets_root_path, is_file
from omni.isaac.core.utils.extensions import enable_extension
from Bot_Control import RLBotController

# CARTER

class RL_Bot():
    def __init__(self, world, simulation_app):

        self.world = world
        self.kit = simulation_app

        self.rl_bot_asset_path = assets_root_path + "/Isaac/Robots/Carter/nova_carter_sensors.usd"
        self.rl_bot_carter =WheeledRobot(    
            prim_path="/World/Nova_Carter",
            name="Carter",
            wheel_dof_names=["joint_caster_left", "joint_caster_right", "joint_wheel_left", "joint_wheel_right"],
            wheel_dof_indices=[3, 4, 5, 6],
            create_robot=True,
            usd_path=rl_bot_asset_path,
            position=np.array([0.4, -0.4,0]))
        self.world.scene.add(carter_bot) 
        self.rl_bot = ArticulationView(prim_paths_expr="/World/Nova_Carter", name="Carter")
        self.rl_bot = self.world.scene.get_object("Carter")
        print(f"Carter : {self.rl_bot}")
        print("Carter in the World")

# CARTER CAMERA
        self.rl_bot_camera = Camera(prim_path="/World/Nova_Carter/chassis_link/front_owl/camera",
                                    name='Carter_Camera',
                                    frequency=30,
                                    resolution=(512,512))
        self.rl_bot_camera.initialize()
        self.kit.update()
        self.rl_bot_camera.initialize()
        print(f"Carter Camera : {carter_camera}")
        self.world.initialize_physics()

# CARTER LiDAR
        self.rl_bot_lidar = self.world.scene.add(
            LidarRtx(prim_path="/World/Nova_Carter/chassis_link/front_RPLidar/RPLIDAR_S2E", 
                    name="Carter_Lidar"))
        print(f"Carter LiDAR : {self.rl_bot_lidar}")
        print("performing reset")
        self.world.reset()
        print("reset done")
        self.rl_bot_lidar.add_range_data_to_frame()
        self.rl_bot_lidar.add_point_cloud_data_to_frame()
        self.rl_bot_lidar.enable_visualization()

# CARTER CONTROLLER
        self.rl_bot_controller = RLBotController()
        print("Controller Created!")