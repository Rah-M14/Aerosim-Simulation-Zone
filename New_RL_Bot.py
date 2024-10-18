
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.sensor import Camera
from omni.isaac.sensor import LidarRtx
import omni.kit.commands
import omni.replicator.core as rep

import asyncio
from scipy import signal
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from pxr import Gf
import numpy as np
import random

class RLBot():
    def __init__(self, simulation_app, world, timeline, assets_root_path, botname):

        if botname.lower() == "carter":
            self.kit = simulation_app
            self.world = world
            self.timeline = timeline
            self.assets_root_path = assets_root_path

            # CARTER
            self.rl_bot_asset_path = self.assets_root_path + "/Isaac/Robots/Carter/carter_v1.usd"
            self.rl_bot_carter = WheeledRobot(    
                prim_path="/World/carter",
                name="RL_Bot",
                wheel_dof_names=["left_wheel", "right_wheel"],
                wheel_dof_indices=[0, 1],
                create_robot=True,
                usd_path=self.rl_bot_asset_path,
                position=np.array([0.4, -0.4,0]))
            self.world.scene.add(self.rl_bot_carter) 

            self.rl_bot = ArticulationView(prim_paths_expr="/World/carter", name="RL_Bot")
            self.rl_bot = self.world.scene.get_object("RL_Bot")
            print(f"RL_Bot : {self.rl_bot}")
            print("RL Bot in the World")

            # CARTER CAMERA
            self.rl_bot_camera = Camera(prim_path="/World/carter/chassis_link/camera_mount/carter_camera_first_person",
                                        name='Carter_Camera',
                                        frequency=30,
                                        resolution=(512,512))
            self.rl_bot_camera.initialize()
            self.kit.update()
            self.rl_bot_camera.initialize()
            print(f"RL_Bot Camera : {self.rl_bot_camera}")
            self.world.initialize_physics()

            lidar_config = "SocEnv_Lidar"

            # CARTER LIDAR
            _, self.rl_bot_lidar = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path="/XT_32_10hz",
                parent="/World/carter/chassis_link",
                config=lidar_config,
                translation=(0, 0, 0.5),
                orientation=Gf.Quatd(1,0,0,0),
            )
            render_product = rep.create.render_product(self.rl_bot_lidar.GetPath(), [1, 1])

            self.rl_bot_lidar.annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
            self.rl_bot_lidar.annotator.attach(render_product)

            writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
            writer.attach(render_product)
            
        elif botname.lower() == "jackal":
                
            self.kit = simulation_app
            self.world = world
            self.timeline = timeline
            self.assets_root_path = assets_root_path

            # JACKAL
            self.rl_bot_asset_path = self.assets_root_path + "/Isaac/Robots/Clearpath/Jackal/jackal.usd"
            self.rl_bot_carter = WheeledRobot(    
                prim_path="/World/jackal",
                name="RL_Bot",
                wheel_dof_names=["front_left_wheel_joint", "front_right_wheel_joint", "rear_left_wheel_joint", "rear_right_wheel_joint"],
                wheel_dof_indices=[0, 1, 2, 3],
                create_robot=True,
                usd_path=self.rl_bot_asset_path,
                position=np.array([0.4, -0.4,0]))
            self.world.scene.add(self.rl_bot_carter) 

            self.rl_bot = ArticulationView(prim_paths_expr="/World/jackal", name="RL_Bot")
            self.rl_bot = self.world.scene.get_object("RL_Bot")
            print(f"RL_Bot : {self.rl_bot}")
            print("RL Bot in the World")

            # JACKAL CAMERA
            self.rl_bot_camera = Camera(prim_path="/World/jackal/base_link/bumblebee_stereo_camera_frame/bumblebee_stereo_left_frame/bumblebee_stereo_left_camera",
                                        name='Jackal_Camera',
                                        frequency=30,
                                        resolution=(512,512))
            self.rl_bot_camera.initialize()
            self.kit.update()
            self.rl_bot_camera.initialize()
            print(f"RL_Bot Camera : {self.rl_bot_camera}")
            self.world.initialize_physics()

            lidar_config = "SocEnv_Lidar"

            # JACKAL LIDAR
            _, self.rl_bot_lidar = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                path="/Lidar",
                parent="/World/jackal/base_link/sick_lms1xx_lidar_frame",
                config=lidar_config,
                translation=(0, 0, 0.033),
                orientation=Gf.Quatd(1,0,0,0),
            )
            render_product = rep.create.render_product(self.rl_bot_lidar.GetPath(), [1, 1])

            self.rl_bot_lidar.annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
            self.rl_bot_lidar.annotator.attach(render_product)

            writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
            writer.attach(render_product)

        elif botname.lower() == "nova_carter":
                
            self.kit = simulation_app
            self.world = world
            self.timeline = timeline
            self.assets_root_path = assets_root_path

            # NOVA CARTER
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

            # NOVA CARTER CAMERA
            self.rl_bot_camera = Camera(prim_path="/World/Nova_Carter/chassis_link/front_hawk/left/camera_left",
                                        name='Carter_Camera',
                                        frequency=30,
                                        resolution=(512,512))
            self.rl_bot_camera.initialize()
            self.kit.update()
            self.rl_bot_camera.initialize()
            print(f"RL_Bot Camera : {self.rl_bot_camera}")
            self.world.initialize_physics()

            lidar_config = "SocEnv_Lidar"

            # NOVA CARTER LIDAR
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


    def _get_lidar_data(self):
        self.kit.update()
        self.timeline.pause()
        rep.orchestrator.step()
        self.rl_bot_lidar.data = self.rl_bot_lidar.annotator.get_data()
        self.timeline.play()
        self.kit.update()
        return self.rl_bot_lidar.data['data']

    def get_denoised_lidar_data(self):
        raw_data = self._get_lidar_data()        
        point_cloud = np.array(raw_data)
        
        if point_cloud.size == 0:
            print("Warning: Empty LiDAR data")
            return np.array([])

        if point_cloud.ndim == 1:
            point_cloud = point_cloud.reshape(-1, 1)
        elif point_cloud.ndim > 2:
            point_cloud = point_cloud.reshape(-1, point_cloud.shape[-1])

        window_size = 5
        smoothed_cloud = np.apply_along_axis(lambda x: signal.medfilt(x, kernel_size=window_size), 0, point_cloud)

        if smoothed_cloud.shape[0] > 5:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(smoothed_cloud)
            denoised_cloud = smoothed_cloud[clusters != -1]
        else:
            denoised_cloud = smoothed_cloud

        return denoised_cloud

    def bot_reset(self):
        valid_pos_x = random.choice(list(set([x for x in np.linspace(-7.5, 7.6, 10000)]) - set(y for y in np.append(np.linspace(-2.6,-1.7,900), np.append(np.linspace(-0.8,0.4,1200), np.append(np.linspace(1.5,2.4,900), np.linspace(3.4,4.6,1200)))))))
        valid_pos_y = random.choice(list(set([x for x in np.linspace(-5.5, 5.6, 14000)]) - set(y for y in np.append(np.linspace(-1.5,2.5,1000), np.linspace(-2.5,-5.6,3100)))))
        new_pos = np.array([valid_pos_x, valid_pos_y, 0.0])

        yaw_angle = random.uniform(0, 2 * np.pi)
        orientation = np.array([np.cos(yaw_angle / 2), 0, 0, np.sin(yaw_angle / 2)])  # Quaternion for rotation around z-axis
        # Set the bot's state with the new position and orientation
        self.rl_bot.set_default_state(position=new_pos, orientation=orientation)
        print("Bot is Reset!")
    