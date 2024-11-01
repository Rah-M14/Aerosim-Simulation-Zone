
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core import SimulationContext
from omni.isaac.sensor import Camera
from omni.isaac.sensor import LidarRtx
from omni.isaac.range_sensor._range_sensor import acquire_lidar_sensor_interface
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils import stage
import omni.kit.commands
import omni.replicator.core as rep
import omni.kit.viewport.utility
import asyncio  
# from RL_Bot_Control import RLBotController

from pxr import Gf
import numpy as np
import random
import omni
import carb

# Nova Carter In our Case
class RLBot():
    def __init__(self, world, simulation_app, timeline, assets_root_path):

        self.kit = simulation_app
        self.world = world
        self.timeline = timeline
        self.assets_root_path = assets_root_path
        self.lidar_path = None

        self.rl_bot_asset_path = self.assets_root_path + "/Isaac/Robots/Carter/nova_carter_sensors.usd"
        self.rl_bot_carter = WheeledRobot(    
            prim_path="/World/Nova_Carter",
            name="RL_Bot",
            wheel_dof_names=["joint_caster_left", "joint_caster_right", "joint_wheel_left", "joint_wheel_right"],
            wheel_dof_indices=[3, 4, 5, 6],
            create_robot=True,
            usd_path=self.rl_bot_asset_path,
            position=np.array([0.4, -0.4,0]))
        # self.rl_bot_carter.initialize()
        # self.rl_bot_carter.set_enabled_self_collisions(True)
        self.world.scene.add(self.rl_bot_carter) 

        self.rl_bot = ArticulationView(prim_paths_expr="/World/Nova_Carter", name="RL_Bot")
        # self.rl_bot.initialize()
        self.rl_bot = self.world.scene.get_object("RL_Bot")
        # self.rl_bot.initialize()
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
        self.rl_bot_lrtx = LidarRtx(prim_path="/World/Nova_Carter/chassis_link/front_RPLidar/RPLIDAR_S2E", 
                    name="Carter_Lidar")
        self.rl_bot_lidar = self.world.scene.add(self.rl_bot_lrtx)
        print(f"RL_Bot LiDAR : {self.rl_bot_lidar}")
        print(type(self.rl_bot_lidar))
        print(f"lrtx : {self.rl_bot_lrtx}")
        print(type(self.rl_bot_lrtx))
        print("performing reset")
        self.world.reset()
        print("reset done")
        # self.rl_bot_lrtx.set_dt(0.1)
        # self.rl_bot_lrtx.set_frequency(10.0)
        self.rl_bot_lrtx.add_range_data_to_frame()
        self.rl_bot_lrtx.add_point_cloud_data_to_frame()
        self.rl_bot_lrtx.enable_visualization()
        print("RL Bot Initialized!")
        
        # # LIDAR
        # self.lsi = acquire_lidar_sensor_interface()
        # result, self.rl_bot_lidar = omni.kit.commands.execute(
        #                 "RangeSensorCreateLidar",
        #                 path="/Nova_Carter/chassis_link/XT_32/RPLIDAR_Req",
        #                 parent="/World",
        #                 min_range=0.4,
        #                 max_range=5.0,
        #                 draw_points=True,
        #                 draw_lines=True,
        #                 horizontal_fov=360.0,
        #                 vertical_fov=30.0,
        #                 horizontal_resolution=1.0,
        #                 vertical_resolution=4.0,
        #                 rotation_rate=100.0,
        #                 high_lod=False,
        #                 yaw_offset=0.0,
        #                 enable_semantics=False)
        
        # print(f"Lidar Created - Result : {result}")        
        # self.lidar_path = str(self.rl_bot_lidar.GetPath())
        # self.lsi = acquire_lidar_sensor_interface(plugin_name="omni.isaac.range_sensor.plugin")

        # # RTX LIDAR
        # lidar_config = "Soc_Lidar"
        # # 1. Create The Camera
        # _, self.rl_bot_lidar = omni.kit.commands.execute(
        #     "IsaacSensorCreateRtxLidar",
        #     path="/PandarXT_32_10hz",
        #     parent="/World/Nova_Carter/chassis_link/XT_32",
        #     config=lidar_config,
        #     translation=(0, 0, 0.0),
        #     orientation=Gf.Quatd(1,0,0,0),
        # )
        # render_product = rep.create.render_product(self.rl_bot_lidar.GetPath(), [1, 1])

        # annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
        # annotator.attach(render_product)

        # writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
        # writer.attach(render_product)
        # # self.rl_bot_lidar.add_range_data_to_frame()
        # # self.rl_bot_lidar.add_point_cloud_data_to_frame()
        # # self.rl_bot_lidar.enable_visualization()

    def get_lidar_feed(self):
        self.kit.update()
        self.timeline.pause()
        point_cloud = self.lsi.get_point_cloud_data(self.lidar_path)
        depth = self.lsi.get_linear_depth_data(self.lidar_path)
        azimuth = self.lsi.get_azimuth_data(self.lidar_path)
        intensity = self.lsi.get_intensity_data(self.lidar_path)
        linear_depth = self.lsi.get_linear_depth_data(self.lidar_path)
        num_rows = self.lsi.get_num_rows(self.lidar_path)
        num_cols_ticked = self.lsi.get_num_cols_ticked(self.lidar_path)
        self.timeline.play()
        self.kit.update()
        return [point_cloud, (depth, azimuth, linear_depth, intensity, num_rows, num_cols_ticked)]

    # async def run_lidar_instance(self):
    #     self.timeline.play()
    #     # lidar_data = asyncio.ensure_future(self.get_lidar_param())
    #     lidar_data = await self.get_lidar_param()
    #     print("Lidar Data : ", lidar_data)
    #     return lidar_data
    
    def bot_reset(self):
        valid_pos_x = random.choice(list(set([x for x in np.linspace(-7.5, 7.6, 10000)]) - set(y for y in np.append(np.linspace(-2.6,-1.7,900), np.append(np.linspace(-0.8,0.4,1200), np.append(np.linspace(1.5,2.4,900), np.linspace(3.4,4.6,1200)))))))
        valid_pos_y = random.choice(list(set([x for x in np.linspace(-5.5, 5.6, 14000)]) - set(y for y in np.append(np.linspace(-1.5,2.5,1000), np.linspace(-2.5,-5.6,3100)))))
        new_pos = np.array([valid_pos_x, valid_pos_y, 0.0])

        self.rl_bot.set_default_state(position=new_pos, orientation=np.array([1, 0, 0, 0]))
        # self.world.reset()
        print("Bot is reset!")