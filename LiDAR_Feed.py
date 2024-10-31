import numpy as np
import contextlib
from ultralytics import YOLO
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from sklearn.cluster import AgglomerativeClustering, OPTICS, DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment

def create_fov_polygon_in_robot_frame(fov_angle=108, fov_radius=5.5):
    angle_start = -fov_angle / 2  # start angle in degrees
    angle_end = fov_angle / 2  # end angle in degrees
    angles = np.linspace(np.radians(angle_start), np.radians(angle_end), num=50)
    # Create points representing the FOV arc in local robot coordinates
    fov_points = [(fov_radius * np.cos(a), fov_radius * np.sin(a)) for a in angles]
    fov_polygon = Polygon(
        [(0, 0)] + fov_points + [(0, 0)]
    )  # close the polygon at origin
    return fov_polygon

def transform_fov_polygon_to_world(fov_polygon, bot_pos_r, R):
    """
    Transforms the FOV polygon from the robot's frame to the world frame using
    the robot's rotation matrix and position.
    """
    transformed_coords = []
    for x, y in fov_polygon.exterior.coords:
        # Apply rotation to each point in the FOV
        rotated_point = np.dot(np.array([x, y, 0]), R.T)
        # Translate by robot's position in world frame
        world_point = rotated_point[:2] + bot_pos_r[:2]
        transformed_coords.append(world_point)
    transformed_coords = np.array(transformed_coords)
    return Polygon(transformed_coords)

def extract_bounding_boxes(yolo_model, frame):
    height, width, c = frame.shape
    # Access the class names directly from the YOLO model
    class_names = yolo_model.model.names  # Correctly access class names from the model
    with contextlib.redirect_stdout(None):
        results = yolo_model(frame, verbose=False)
    # Extract bounding boxes in xyxy format
    boxes = results[0].boxes.xyxy.cpu().numpy()
    # Extract class indices
    class_indices = results[0].boxes.cls.cpu().numpy()
    frame_boxes = []
    frame_labels = []
    for box, cls_id in zip(boxes, class_indices):
        x1, y1, x2, y2 = map(int, box[:4])
        frame_boxes.append((x1 / width, y1 / height, x2 / width, y2 / height))
        frame_labels.append(class_names[int(cls_id)])  # Map class index to class name
    return frame_boxes, frame_labels

def bounding_box_to_lidar_projection(
    bbox, robot_position, fov_degrees=108, distance_scaling_factor=1.5
):
    """
    Projects a camera bounding box into LiDAR 2D space and translates it based on the robot's position.
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    # Convert normalized x_center to an angle in LiDAR's 2D plane
    fov_radians = np.radians(fov_degrees)
    angle = (x_center - 0.5) * fov_radians  # Map x_center to [-fov/2, +fov/2]
    # Estimate a distance (manually tuned with a scaling factor)
    bbox_height = y_max - y_min
    distance = distance_scaling_factor * (1 / bbox_height)
    # Convert polar coordinates (angle, distance) to Cartesian (x, y)
    x_lidar = distance * np.cos(angle)
    y_lidar = distance * np.sin(angle)
    # Translate by the robot's position
    x_lidar += robot_position[0]
    y_lidar += robot_position[1]
    # Create a simple polygon around this point (to simulate object extent)
    extent_w = y_max - y_min
    extent_h = x_max - x_min
    polygon = Polygon(
        [
            (x_lidar - extent_w, -1 * (y_lidar - extent_h)),
            (x_lidar + extent_w, -1 * (y_lidar - extent_h)),
            (x_lidar + extent_w, -1 * (y_lidar + extent_h)),
            (x_lidar - extent_w, -1 * (y_lidar + extent_h)),
        ]
    )
    return polygon
# Optional: Toggle projection of bounding boxes into LiDAR space

def get_bounding_box_centroids(
    bounding_boxes, project_to_lidar=True, robot_position=(0, 0)
):
    """
    Get bounding box centroids, with an option to project into LiDAR space.
    """
    if project_to_lidar:
        return [
            bounding_box_to_lidar_projection(bbox, robot_position).centroid.coords[0]
            for bbox in bounding_boxes
        ]
    else:
        # Return bounding box centroids in the original space
        return [
            [(x_min + x_max) / 2, (y_min + y_max) / 2]
            for x_min, y_min, x_max, y_max in bounding_boxes
        ]
# Function to compute pairwise distance matrix

def compute_distance_matrix(centroids):
    return pairwise_distances(centroids, metric="euclidean")

def assign_labels_to_clusters(bbox_centroids, cluster_centroids, labels):
    """
    Assigns bounding boxes to clusters based on spatial similarity using the Hungarian Algorithm.
    The assignment is done at the cluster level.
    Args:
        bbox_centroids: List of bounding box centroids.
        cluster_centroids: List of cluster centroids.
        labels: List of labels corresponding to the bounding boxes.
    Returns:
        A dictionary where the key is the cluster label and the value is the assigned bounding box label.
    """
    n_boxes = len(bbox_centroids)
    n_clusters = len(cluster_centroids)
    if n_boxes > n_clusters:
        # raise ValueError("More bounding boxes than clusters, cannot assign!")
        return {}
    # Compute the pairwise distance matrix
    if n_boxes == 0 or n_clusters == 0:
        return {}
    
    distance_matrix = pairwise_distances(
        bbox_centroids, cluster_centroids, metric="euclidean"
    )
    # Solve the assignment problem using the Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(distance_matrix)
    # Create a mapping of cluster index to bounding box label
    cluster_label_map = {
        col_indices[i]: labels[row_indices[i]] for i in range(len(row_indices))
    }
    return cluster_label_map
# Function to assign a label to each point in the LiDAR data

def assign_labels_to_points(lidar_points, labels_clusters, cluster_label_map):
    """
    Assigns labels to each point in the LiDAR data based on the cluster it belongs to.
    Args:
        lidar_points: LiDAR points data.
        labels_clusters: Cluster labels for each LiDAR point.
        cluster_label_map: Mapping of cluster index to bounding box label.
    Returns:
        A list of labels for each LiDAR point.
    """
    point_labels = []

    for cluster_label in labels_clusters:
        if cluster_label != -1:  # Ignore noise points
            if cluster_label not in cluster_label_map:
                point_labels.append("Noise")
            else:
                point_labels.append(cluster_label_map[cluster_label])
        else:
            point_labels.append("Noise")
    return point_labels

def get_cluster_centroid(cluster_points):
    return np.mean(cluster_points, axis=0)

def quaternion_to_rotation_matrix(ori):
    w, x, y, z = ori
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )

def get_licam_image(lidar_data, camera_image, mlp_obs, yolo_model, world_min_max, project_camera=True, image_size=64):
    if lidar_data.size == 0:
        return None, None
    pos = mlp_obs[0]
    ori = mlp_obs[1]
    goal_pos = mlp_obs[-2]

    quat_matrix = quaternion_to_rotation_matrix(ori)
    rotated_data = np.dot(lidar_data[:, :2], quat_matrix[:2, :2].T)
    world_data = rotated_data + pos
    fov_polygon_robot = create_fov_polygon_in_robot_frame(fov_angle=108)
    # Transform the FOV polygon to the world frame using the robot's orientation
    fov_polygon_world = transform_fov_polygon_to_world(fov_polygon_robot, pos, quat_matrix)
    # Filter LiDAR points within camera FOV
    filtered_points = [
        p for p in world_data if fov_polygon_world.contains(Point(p[0], p[1]))
    ]
    lidar_points = np.array(filtered_points)

    if lidar_points.size == 0:
        return np.zeros((3, image_size, image_size), dtype=np.uint8) / 255.0

    # yolo_model = YOLO("yolov9t.pt")
    bounding_boxes, labels_bbox = extract_bounding_boxes(yolo_model, camera_image)

    # if lidar_points.size < len(bounding_boxes) or lidar_points.size < len(labels_bbox):
    #     return np.zeros((3, image_size, image_size), dtype=np.uint8) / 255.0

    projected_lidar_polygons = [
        bounding_box_to_lidar_projection(bbox, pos) for bbox in bounding_boxes
    ]
    clustering_model = KMeans(n_clusters=len(bounding_boxes) if len(bounding_boxes) > 0 else 1)

    if lidar_points.shape[0] < len(bounding_boxes):
        return np.zeros((3, image_size, image_size), dtype=np.uint8) / 255.0
    else:
        labels_clusters = clustering_model.fit_predict(lidar_points)
    
    project_to_lidar = project_camera

    # Get bounding box centroids (either in original space or projected)
    bbox_centroids = get_bounding_box_centroids(
        bounding_boxes, project_to_lidar, pos
    )
    cluster_centroids = []
    for label in np.unique(labels_clusters):
        if label != -1:  # Ignore noise points labeled as -1
            cluster_points = lidar_points[labels_clusters == label]
            cluster_centroids.append(get_cluster_centroid(cluster_points))
    cluster_label_map = assign_labels_to_clusters(
        bbox_centroids, cluster_centroids, labels_bbox
    )
    point_labels = assign_labels_to_points(
        lidar_points, labels_clusters, cluster_label_map
    )
    lidar_points[:, 0] = (lidar_points[:, 0] - world_min_max[0]) / (
        world_min_max[2] - world_min_max[0]
    )
    lidar_points[:, 1] = (lidar_points[:, 1] - world_min_max[1]) / (
        world_min_max[3] - world_min_max[1]
    )
    world_data[:, 0] = (world_data[:, 0] - world_min_max[0]) / (
        world_min_max[2] - world_min_max[0]
    )
    world_data[:, 1] = (world_data[:, 1] - world_min_max[1]) / (
        world_min_max[3] - world_min_max[1]
    )
    normalized_pos = (pos[:2] - world_min_max[:2]) / (
        world_min_max[2:] - world_min_max[:2]
    )
    normalized_goal = (goal_pos[:2] - world_min_max[:2]) / (
        world_min_max[2:] - world_min_max[:2]
    )
    # Create a blank image
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    # Convert world_data and lidar_points to pixel coordinates
    pixel_coords = (world_data * (image_size - 1)).astype(int)
    lidar_pixel_coords = (lidar_points * (image_size - 1)).astype(int)
    bot_pixel = (normalized_pos * (image_size - 1)).astype(int)
    goal_pixel = (normalized_goal * (image_size - 1)).astype(int)
    # Filter valid pixels (those inside image bounds)
    valid_pixels = (
        (pixel_coords[:, 0] >= 0)
        & (pixel_coords[:, 0] < image_size)
        & (pixel_coords[:, 1] >= 0)
        & (pixel_coords[:, 1] < image_size)
    )
    valid_pixel_coords = pixel_coords[valid_pixels]
    # Color map for different labels
    label_color_map = {
        "default": [255, 0, 0],  # Red
        "person": [0, 255, 0],  # Green
        "on camera_other": [0, 0, 255],  # Blue
        "bot": [255, 255, 255],  # White for bot
        "goal": [255, 255, 0] # Yellow for Goal
    }
    # Assign default color (red) to all valid points
    image[valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]] = label_color_map[
        "default"
    ]
    # Recolor points that are part of lidar_points based on labels
    valid_lidar_pixels = (
        (lidar_pixel_coords[:, 0] >= 0)
        & (lidar_pixel_coords[:, 0] < image_size)
        & (lidar_pixel_coords[:, 1] >= 0)
        & (lidar_pixel_coords[:, 1] < image_size)
    )
    valid_lidar_coords = lidar_pixel_coords[valid_lidar_pixels]
    # valid_labels = point_labels[valid_lidar_pixels]
    # Loop through lidar points and assign color based on the label
    for idx, (x, y) in enumerate(valid_lidar_coords):
        label = point_labels[idx]  # Get the label for the current lidar point
        if label in label_color_map:
            color = label_color_map[label]
        else:
            color = label_color_map["on camera_other"]
        image[y, x] = color
    # Mark the robot position (bot) as a white square
    bot_size = 1
    bot_x, bot_y = bot_pixel
    image[
        max(0, bot_y - bot_size) : min(image_size, bot_y + bot_size + 1),
        max(0, bot_x - bot_size) : min(image_size, bot_x + bot_size + 1),
    ] = label_color_map[
        "bot"
    ]  # White square for bot

    goal_x, goal_y = goal_pixel
    image[
        max(0, int(goal_y) - bot_size) : min(image_size, int(goal_y) + bot_size + 1),
        max(0, int(goal_x) - bot_size) : min(image_size, int(goal_x) + bot_size + 1),
    ] = label_color_map[
        "goal"
    ]  # Yellow Square for Goal

    # np.save("/home/rah_m/TEST/Sample_image.npy",image)
    
    image = np.swapaxes(image,0,-1)
    
    return image/255.0