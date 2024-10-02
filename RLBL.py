
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
        #             name="Carter_Lidar", config="RPLIDAR_S2E_Soc_Nav"))
        # print(f"RL_Bot LiDAR : {self.rl_bot_lidar}")
        # print("performing reset")
        # self.world.reset()
        # print("reset done")
        # self.rl_bot_lidar.add_range_data_to_frame()
        # self.rl_bot_lidar.add_point_cloud_data_to_frame()
        # self.rl_bot_lidar.enable_visualization()
        # print("RL Bot Initialized!")
        
        # LIDAR

        # from omni.isaac.range_sensor import 
        lidar_config = "RPLIDAR_S2E_Soc_Nav"
        enable_extension("omni.isaac.debug_draw")

        self.lsi = acquire_lidar_sensor_interface()

        result, self.rl_bot_lidar = omni.kit.commands.execute(
                        "RangeSensorCreateLidar",
                        path="/Nova_Carter/chassis_link/XT_32/RPLIDAR_Req",
                        parent="/World",
                        min_range=0.4,
                        max_range=10.0,
                        draw_points=True,
                        draw_lines=True,
                        horizontal_fov=360.0,
                        vertical_fov=30.0,
                        horizontal_resolution=0.4,
                        vertical_resolution=4.0,
                        rotation_rate=20.0,
                        high_lod=False,
                        yaw_offset=0.0,
                        enable_semantics=False)
        
        print(f"Lidar Created - Result : {result}")        
        self.lidar_path = str(self.rl_bot_lidar.GetPath())
        self.lsi = acquire_lidar_sensor_interface(plugin_name="omni.isaac.range_sensor.plugin")

        # print(f"path of lidar : {self.lidar_path}")
        # print(type(self.lidar_path))

        # if self.lidar_path == "/World/Nova_Carter/chassis_link/XT_32/RPLIDAR_Req":
        #     print("Lidar Path is valid")

        # print(f"Type of lsi : {type(self.lsi)}")
        # print("lidar sensor interface acquired")

        # if self.lsi.is_lidar_sensor(self.lidar_path):
        #     print("lidar sensor is valid")

    # async def get_lidar_param(self):
    #         await omni.kit.app.get_app().next_update_async()
    #         self.timeline.pause()
    #         depth = self.lsi.get_linear_depth_data(self.lidar_path)
    #         azimuth = self.lsi.get_azimuth_data(self.lidar_path)
    #         intensity = self.lsi.get_intensity_data(self.lidar_path)
    #         linear_depth = self.lsi.get_linear_depth_data(self.lidar_path)
    #         self.timeline.play()
    #         return (depth, azimuth, intensity, linear_depth)

    def get_lidar_param(self):
        self.kit.update()
        self.timeline.pause()
        depth = self.lsi.get_linear_depth_data(self.lidar_path)
        azimuth = self.lsi.get_azimuth_data(self.lidar_path)
        intensity = self.lsi.get_intensity_data(self.lidar_path)
        linear_depth = self.lsi.get_linear_depth_data(self.lidar_path)
        self.timeline.play()
        self.kit.update()
        return (depth, azimuth, intensity, linear_depth)

    async def run_lidar_instance(self):
        self.timeline.play()
        # lidar_data = asyncio.ensure_future(self.get_lidar_param())
        lidar_data = await self.get_lidar_param()
        print("Lidar Data : ", lidar_data)
        return lidar_data
        
        # # if self.rl_bot_lidar_interface.is_lidar_sensor(lidar_path):
        # #     print("lidar sensor is valid")

        # print("Lidar Created!")
        # # depth = self.lsi.get_linear_depth_data(lidar_path)

        # # dep_data = self.lsi.get_depth_data("/World/Nova_Carter/chassis_link/front_RPLidar/RPLIDAR_S2E_01")
        # # print(f"Depth Data : {depth}")

        # # 1. Create The Lidar
        # _, self.rl_bot_lidar = omni.kit.commands.execute(
        #     "IsaacSensorCreateRtxLidar",
        #     path="/RPLIDAR_S2E",
        #     parent="/World/Nova_Carter/chassis_link/front_RPLidar",
        #     config=lidar_config,
        #     translation=(0, 0, 0.0),
        #     orientation=Gf.Quatd(1,0,0,0),

        # )

        # print(f"RL_Bot_Lidar Path : {self.rl_bot_lidar.GetPath()}")

        # hydra_texture = rep.create.render_product(self.rl_bot_lidar.GetPath(), [1, 1], name="Isaac")
        # simulation_context = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
        # self.kit.update()

        # # 2. Create and Attach a render product to the camera
        # render_product = rep.create.render_product(self.rl_bot_lidar.GetPath(), [1, 1])

        # # 3. Create Annotator to read the data from with annotator.get_data()
        # annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
        # annotator.initialize(outputTimestamp=True)
        # annotator.attach(render_product)

        # # 4. Create a Replicator Writer that "writes" points into the scene for debug viewing
        # # writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
        # # writer.initialize(output_dir=f"/home/rah_m/Isaac_World_Files/")
        # # writer.attach(render_product)

        # writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloud" + "Buffer")

        # # writer.initialize(output_dir=f"/home/rah_m/Isaac_World_Files/")
        # writer.attach([hydra_texture])

        # self.kit.update()
        # simulation_context.play()

        # Bot Parameters
        # self.rl_bot.current_state = self.rl_bot.get_world_pose()

    def bot_reset(self):
        valid_pos_x = random.choice(list(set([x for x in np.linspace(-7.5, 7.6, 10000)]) - set(y for y in np.append(np.linspace(-2.6,-1.7,900), np.append(np.linspace(-0.8,0.4,1200), np.append(np.linspace(1.5,2.4,900), np.linspace(3.4,4.6,1200)))))))
        valid_pos_y = random.choice(list(set([x for x in np.linspace(-5.5, 5.6, 14000)]) - set(y for y in np.append(np.linspace(-1.5,2.5,1000), np.linspace(-2.5,-5.6,3100)))))
        new_pos = np.array([valid_pos_x, valid_pos_y, 0.0])

        self.rl_bot.set_default_state(position=new_pos, orientation=np.array([1, 0, 0, 0]))
        # self.world.reset()
        print("Bot is reset!")