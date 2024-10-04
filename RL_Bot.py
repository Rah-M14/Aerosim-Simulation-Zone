
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

        # CARTER LIDAR
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

    # def get_denoised_lidar_data(self):
    #     try:
    #         raw_data = self._get_lidar_data()
    #         if len(raw_data) == 0:
    #             print("Warning: Empty LiDAR data")
    #             return np.array([])
                
    #         point_cloud = np.array(raw_data)
            
    #         filtered_cloud = self._combined_filter(point_cloud)
    #         if len(filtered_cloud) == 0:
    #             print("Warning: No points left after combined filtering")
    #             return np.array([])

    #         non_ground_filtered_points = self._remove_ground_and_cluster(filtered_cloud)
    #         if len(non_ground_filtered_points) == 0:
    #             print("Warning: No non-ground points found after clustering")
    #             return np.array([])

    #         return non_ground_filtered_points

    #     except Exception as e:
    #         print(f"Error in get_denoised_lidar_data: {str(e)}")
    #         return np.array([])

    # def _combined_filter(self, points, min_distance=0.05, max_distance=5.0, voxel_size=0.05, k=50, z_max=2.0):
    #     if len(points) == 0:
    #         return np.array([])

    #     # Voxel grid filter
    #     voxel_indices = np.floor(points / voxel_size).astype(int)
    #     _, inverse, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)
    #     accumulated_points = np.zeros((len(counts), 3))
    #     np.add.at(accumulated_points, inverse, points)
    #     voxel_filtered = accumulated_points / counts[:, np.newaxis]

    #     if len(voxel_filtered) <= 1:
    #         return voxel_filtered

    #     # Statistical outlier removal
    #     tree = cKDTree(voxel_filtered)
    #     distances, _ = tree.query(voxel_filtered, k=min(k, len(voxel_filtered)-1))
    #     mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self-distance
    #     threshold = np.mean(mean_distances) + z_max * np.std(mean_distances)
    #     return voxel_filtered[mean_distances < threshold]

    # def _remove_ground_and_cluster(self, points, height_threshold=0.1, max_slope=0.1, eps=0.1, min_samples=5):
    #     sorted_z = np.sort(points[:, 2])
    #     ground_level = np.median(sorted_z[:min(100, len(sorted_z))])
    #     non_ground_mask = points[:, 2] > (ground_level + height_threshold)
        
    #     # Slope-based filtering for points close to the ground
    #     close_to_ground = ~non_ground_mask
    #     if np.any(close_to_ground):
    #         tree = cKDTree(points[close_to_ground, :2])
    #         for i in np.where(close_to_ground)[0]:
    #             neighbors = tree.query_ball_point(points[i, :2], r=0.5)
    #             if len(neighbors) > 3:
    #                 neighbor_points = points[close_to_ground][neighbors]
    #                 slopes = np.abs(np.polyfit(neighbor_points[:, :2].T, neighbor_points[:, 2], deg=1))
    #                 if np.max(slopes) > max_slope:
    #                     non_ground_mask[i] = True

    #     non_ground_points = points[non_ground_mask]

    #     if len(non_ground_points) <= min_samples:
    #         return non_ground_points

    #     # DBSCAN clustering
    #     db = DBSCAN(eps=eps, min_samples=min(min_samples, len(non_ground_points)-1))
    #     labels = db.fit_predict(non_ground_points)
    #     return non_ground_points[labels != -1]

    # def get_denoised_lidar_data(self):
    #     raw_data = self._get_lidar_data()
    #     point_cloud = np.array(raw_data)
        
    #     # Remove points too close or too far
    #     min_distance = 0.05  # 5 cm
    #     max_distance = 5.0  # 5 meters (as per Soc_Lidar.json)
    #     mask = (point_cloud[:, 2] > min_distance) & (point_cloud[:, 2] < max_distance)
    #     filtered_cloud = point_cloud[mask]

    #     # Apply voxel grid filter
    #     voxel_size = 0.05  # 5 cm voxel size
    #     voxel_filter = self._voxel_grid_filter(filtered_cloud, voxel_size)

    #     # Apply statistical outlier removal
    #     filtered_cloud = self._statistical_outlier_removal(voxel_filter, k=50, z_max=2.0)

    #     # Apply ground plane removal
    #     # non_ground_points = self._remove_ground_plane(filtered_cloud, height_threshold=0.1, max_slope=0.1)

    #     # Apply DBSCAN clustering to remove small clusters (noise)
    #     dbscan = DBSCAN(eps=0.1, min_samples=5)
    #     clusters = dbscan.fit_predict(filtered_cloud)
    #     denoised_cloud = filtered_cloud[clusters != -1]

    #     return denoised_cloud

    # def _voxel_grid_filter(self, points, voxel_size):
    #     voxel_indices = np.floor(points / voxel_size).astype(int)
    #     _, inverse_indices, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)
    #     accumulated_points = np.zeros((len(counts), 3))
    #     np.add.at(accumulated_points, inverse_indices, points)
    #     return accumulated_points / counts[:, np.newaxis]

    # def _statistical_outlier_removal(self, points, k=50, z_max=2.0):
    #     tree = cKDTree(points)
    #     distances, _ = tree.query(points, k=k)
    #     mean_distances = np.mean(distances, axis=1)
    #     std_dev = np.std(mean_distances)
    #     threshold = mean_distances.mean() + z_max * std_dev
    #     mask = mean_distances < threshold
    #     return points[mask]

    # def _remove_ground_plane(self, points, height_threshold=0.1, max_slope=0.1):
    #     # Simple ground plane removal based on height and slope
    #     sorted_points = points[points[:, 2].argsort()]
    #     ground_level = np.median(sorted_points[:100, 2])
        
    #     non_ground_mask = (points[:, 2] > ground_level + height_threshold)
        
    #     # Check slope for points close to the ground
    #     close_to_ground = (points[:, 2] <= ground_level + height_threshold)
    #     for i in range(len(points)):
    #         if close_to_ground[i]:
    #             neighbors = points[np.linalg.norm(points[:, :2] - points[i, :2], axis=1) < 0.5]
    #             if len(neighbors) > 3:
    #                 # Calculate slope using the distance from the point and the height difference
    #                 distances = np.linalg.norm(neighbors[:, :2] - points[i, :2], axis=1)
    #                 height_diffs = neighbors[:, 2] - points[i, 2]
    #                 slopes = np.abs(height_diffs / (distances + 1e-6))  # Add small epsilon to avoid division by zero
    #                 if np.mean(slopes) > max_slope:
    #                     non_ground_mask[i] = True
        
    #     return points[non_ground_mask]

    def get_denoised_lidar_data(self):
        raw_data = self._get_lidar_data()        
        point_cloud = np.array(raw_data)
        
        if point_cloud.size == 0:
            print("Warning: Empty LiDAR data")
            return np.array([])

        # Ensure point_cloud is 2D
        if point_cloud.ndim == 1:
            point_cloud = point_cloud.reshape(-1, 1)
        elif point_cloud.ndim > 2:
            point_cloud = point_cloud.reshape(-1, point_cloud.shape[-1])

        # Apply median filter to smooth out noise
        window_size = 5
        smoothed_cloud = np.apply_along_axis(lambda x: signal.medfilt(x, kernel_size=window_size), 0, point_cloud)

        # Only perform DBSCAN if we have enough points
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

        self.rl_bot.set_default_state(position=new_pos, orientation=np.array([1, 0, 0, 0]))
        print("Bot is reset!")
    