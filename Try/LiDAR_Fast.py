import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np

def get_lidar_points(binary_img, current_pos_batch, world_limits, num_rays=360, max_range=4.0):
    """
    Get LiDAR first contact points for a batch of sensor positions or a single sensor.

    Args:
        binary_img (np.ndarray): Binary image (2D array) where 0 indicates an obstacle and 1 indicates free space.
        current_pos_batch (np.ndarray): Array of shape (B, 2) for batched sensor positions,
                                        or a single sensor's (x,y) coordinates as a 1D array of shape (2,).
        world_limits (np.ndarray): Array of shape (2,2) specifying [[min_x, max_x], [min_y, max_y]] for the world.
        num_rays (int): Number of rays to cast (default 360 for 1° resolution).
        max_range (float): Maximum sensor range in world units.

    Returns:
        contact_points (np.ndarray):
            If batched, an array of shape (B, num_rays, 2) containing the (x,y) relative contact points for each ray.
            If a single sensor is provided, an array of shape (num_rays, 2) is returned.
        lidar_dists (np.ndarray):
            If batched, an array of shape (B, num_rays) containing the Euclidean distance (in world units) for each ray.
            For a single sensor input, an array of shape (num_rays,) is returned.
            Rays that do not hit an obstacle within max_range return a distance of np.inf.
    """
    # Allow single sensor input by adding a batch dimension.
    single_input = False
    if current_pos_batch.ndim == 1:
        current_pos_batch = np.expand_dims(current_pos_batch, axis=0)
        single_input = True

    # Image and world scaling.
    height, width = binary_img.shape
    world_width = world_limits[0][1] - world_limits[0][0]
    world_height = world_limits[1][1] - world_limits[1][0]
    scale_x = width / world_width
    scale_y = height / world_height

    # Batch size.
    B = current_pos_batch.shape[0]

    # Convert world current positions to image coordinates.
    sensor_x = np.rint((current_pos_batch[:, 0] - world_limits[0][0]) * scale_x).astype(np.int32)
    sensor_y = height - np.rint((current_pos_batch[:, 1] - world_limits[1][0]) * scale_y).astype(np.int32)
    sensor_pos_img = np.stack([sensor_x, sensor_y], axis=1)   # shape: (B, 2)

    # Convert max_range from world units to pixels (using the smaller scale factor).
    max_range_px = int(max_range * min(scale_x, scale_y))
    N = max_range_px - 1  # number of steps along each ray

    # Generate all ray angles and their direction vectors (for image coordinates).
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)  # shape: (num_rays,)
    directions = np.stack([np.cos(angles), -np.sin(angles)], axis=0) # shape: (2, num_rays)
    directions_T = directions.T  # shape: (num_rays, 2)

    # Generate distances along the rays.
    ray_lengths = np.arange(1, max_range_px)  # shape: (N,)

    # Compute base ray displacements: shape (N, num_rays, 2)
    base_ray_points = ray_lengths[:, None, None] * directions_T[None, :, :]

    # Compute the pixel coordinates for every ray from every sensor.
    # sensor_pos_img expanded to shape (B, 1, 1, 2) and base_ray_points broadcasted along B.
    sensor_pos_img_expanded = sensor_pos_img[:, None, None, :]  # shape: (B, 1, 1, 2)
    ray_points = sensor_pos_img_expanded + base_ray_points[None, :, :, :]
    ray_points = ray_points.astype(np.int32)  # shape: (B, N, num_rays, 2)

    # Create a validity mask to filter out-of-bound coordinates.
    valid_x = (ray_points[..., 0] >= 0) & (ray_points[..., 0] < width)
    valid_y = (ray_points[..., 1] >= 0) & (ray_points[..., 1] < height)
    valid_mask = valid_x & valid_y   # shape: (B, N, num_rays)

    # Prepare an array for ray pixel values; initialize as free space (1).
    B, N, R, _ = ray_points.shape
    ray_values = np.ones((B, N, R), dtype=binary_img.dtype)
    ray_x = ray_points[..., 0]
    ray_y = ray_points[..., 1]
    ray_values[valid_mask] = binary_img[ray_y[valid_mask], ray_x[valid_mask]]

    # Build an obstacle mask (True where pixel value equals 0).
    obs_mask = (ray_values == 0)  # shape: (B, N, num_rays)

    # For each sensor and ray, determine if an obstacle was detected.
    has_obs = np.any(obs_mask, axis=1)  # shape: (B, num_rays)
    first_obs = np.argmax(obs_mask, axis=1)  # shape: (B, num_rays)
    first_obs[~has_obs] = -1  # Mark rays with no obstacle as -1.

    # Initialize outputs.
    contact_points = np.zeros((B, num_rays, 2), dtype=np.float32)
    lidar_dists = np.ones((B, num_rays), dtype=np.float32) * np.inf

    # Process only those rays where an obstacle was detected.
    valid_rays = has_obs  # shape: (B, num_rays)
    if np.any(valid_rays):
        # Transpose ray_points to (B, num_rays, N, 2) for easier indexing.
        ray_points_perm = np.transpose(ray_points, (0, 2, 1, 3))
        # Expand first_obs indices for proper indexing.
        indices = first_obs[:, :, np.newaxis, np.newaxis]
        obs_pixels = np.take_along_axis(ray_points_perm, indices, axis=2)
        obs_pixels = np.squeeze(obs_pixels, axis=2)  # shape: (B, num_rays, 2)

        # Convert obstacle pixel coordinates back to world coordinates.
        px = obs_pixels[..., 0].astype(np.float32)
        py = obs_pixels[..., 1].astype(np.float32)
        world_x = ((px + 0.5) / scale_x) + world_limits[0][0]
        world_y = world_limits[1][0] + ((height - (py + 0.5)) / scale_y)

        # Compute relative coordinates from the sensor position.
        rel_x = world_x - current_pos_batch[:, 0][:, None]
        rel_y = world_y - current_pos_batch[:, 1][:, None]
        distances = np.sqrt(rel_x**2 + rel_y**2)

        # Only assign valid contact points for detections within max_range.
        within_range = distances <= max_range
        assign_mask = valid_rays & within_range

        contact_stack = np.stack([rel_x, rel_y], axis=-1)
        contact_points[assign_mask, :] = contact_stack[assign_mask]
        lidar_dists[assign_mask] = distances[assign_mask]

    # If a single sensor was provided, remove the batch dimension from the results.
    if single_input:
        contact_points = np.squeeze(contact_points, axis=0)
        lidar_dists = np.squeeze(lidar_dists, axis=0)

    return contact_points, lidar_dists

def create_binary_image(image_path):
    from PIL import Image
    img = np.array(Image.open(image_path).convert('L'))  # Convert to grayscale
    binary_img = (img > 128).astype(np.uint8)  # Threshold to create a binary map
    binary_img = cv2.resize(binary_img, (0,0), fx=0.25, fy=0.25)
    return binary_img

def create_synthetic_binary_image():
    """
    Create a 200x200 synthetic binary image with a square obstacle in the center.
    1 = free space, 0 = obstacle.
    """
    binary_img = np.ones((200, 200), dtype=np.uint8)
    binary_img[90:110, 90:110] = 0
    return binary_img

def main():
    binary_img = create_synthetic_binary_image()
    world_limits = np.array([[-10, 10], [-10, 10]])
    
    # Image dimensions and scaling to convert between world and image coordinates.
    height, width = binary_img.shape
    world_width = world_limits[0][1] - world_limits[0][0]
    world_height = world_limits[1][1] - world_limits[1][0]
    scale_x = width / world_width
    scale_y = height / world_height

    # Define batch sizes to test.
    batch_sizes = [128, 256, 512, 1024, 2048]
    times_taken = []
    sample_result = {}  # To store one sample sensor's result from the first batch.

    for batch in batch_sizes:
        # Generate random sensor positions uniformly in the world limits.
        sensors = np.random.uniform(low=[world_limits[0][0], world_limits[1][0]],
                                    high=[world_limits[0][1], world_limits[1][1]],
                                    size=(batch, 2))
        start_time = time.time()
        contacts, dists = get_lidar_points(binary_img, sensors, world_limits,
                                                   num_rays=360, max_range=4.0)
        elapsed = time.time() - start_time
        times_taken.append(elapsed)
        print(f"Batch size {batch}: Time taken: {elapsed:.6f} seconds")
        
        # For the first batch, save the first sensor result for plotting.
        if batch == batch_sizes[0]:
            sample_result['sensor'] = sensors[0]
            sample_result['contacts'] = contacts[0]

    # Create two subplots: one for LiDAR contacts, and one for the timing plot.
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot sample sensor LiDAR contacts over the binary image.
    axs[0].imshow(binary_img, cmap='gray', origin='upper')
    sensor = sample_result['sensor']
    sensor_img_x = int(round((sensor[0] - world_limits[0][0]) * scale_x))
    sensor_img_y = height - int(round((sensor[1] - world_limits[1][0]) * scale_y))
    axs[0].plot(sensor_img_x, sensor_img_y, 'go', markersize=8, label='Sensor')
    for pt in sample_result['contacts']:
        # Convert the relative contact point to world coordinates.
        world_pt_x = sensor[0] + pt[0]
        world_pt_y = sensor[1] + pt[1]
        img_pt_x = int(round((world_pt_x - world_limits[0][0]) * scale_x))
        img_pt_y = height - int(round((world_pt_y - world_limits[1][0]) * scale_y))
        axs[0].plot(img_pt_x, img_pt_y, 'ro', markersize=2)
    axs[0].set_title("Sample Sensor LiDAR Contacts")
    axs[0].legend()

    # Plot batch size vs. computation time.
    axs[1].plot(batch_sizes, times_taken, marker='o')
    axs[1].set_xlabel("Batch Size")
    axs[1].set_ylabel("Computation Time (seconds)")
    axs[1].set_title("Batched LiDAR Computation Time")
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

###############################################################################
# New functionality: find angles with a LiDAR distance greater than a threshold.
###############################################################################

# def find_angles_with_distance_above_threshold(binary_img, sensor_pos, world_limits, threshold, num_rays=360, max_range=4.0):
#     """
#     Perform a LiDAR scan for a single sensor, and find the scanning angles (in degrees)
#     where the measured LiDAR distance is greater than the given threshold.

#     Args:
#         binary_img (np.ndarray): Binary image (2D array), where 0 indicates obstacle and 1 indicates free space.
#         sensor_pos (np.ndarray): Array-like of shape (2,) for the sensor's (x, y) position in world coordinates.
#         world_limits (np.ndarray): Array with shape (2, 2) defining the world limits as [[min_x, max_x], [min_y, max_y]].
#         threshold (float): The distance threshold in world units.
#         num_rays (int): Number of rays to cast (default is 360).
#         max_range (float): Maximum LiDAR range in world units (default is 4.0).

#     Returns:
#         valid_angles (np.ndarray): Array of angles (in degrees) for which the LiDAR measured distance exceeds the threshold.
#         lidar_dists (np.ndarray): 1D array of LiDAR distances for each ray.
#     """
#     # Run the LiDAR scan for the single sensor.
#     _, lidar_dists = get_lidar_points(binary_img, sensor_pos, world_limits, num_rays=num_rays, max_range=max_range)
    
#     # Generate ray angles in degrees.
#     angles_deg = np.linspace(0, 360, num_rays, endpoint=False)
    
#     # Find all angles where the LiDAR distance is greater than the threshold.
#     valid_indices = np.where(lidar_dists > threshold)[0]
#     valid_angles = angles_deg[valid_indices]
    
#     return valid_angles, lidar_dists

# def demo_find_angles():
#     """
#     Demonstrate the use of find_angles_with_distance_above_threshold.
#     This demo scans the world from a single sensor at (0,0) and highlights
#     the angles where the LiDAR distance is greater than the threshold.
#     """
#     # Create a synthetic binary image with a square obstacle.
#     # binary_img = create_synthetic_binary_image()
#     binary_img = create_binary_image(image_path=r"F:\Aerosim-Simulation-Zone\Try\New_WR_World.png")
#     # Define world limits.
#     world_limits = np.array([[-10, 10], [-8, 8]])
#     # Set the sensor position. Adjust as needed.
#     sensor_pos = np.array([5.0, -3.0])
#     # Define the distance threshold.
#     threshold = 3.0
    
#     valid_angles, lidar_dists = find_angles_with_distance_above_threshold(
#         binary_img, sensor_pos, world_limits, threshold, num_rays=360, max_range=4.0)

#     print("Angles (in degrees) where LiDAR distance is greater than threshold:")
#     print(valid_angles)
    
#     # For visualization, create a polar plot.
#     angles_rad = np.linspace(0, 2*np.pi, 360, endpoint=False)
#     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))
#     ax.plot(angles_rad, lidar_dists, 'b.', label='LiDAR Distances')
    
#     valid_indices = np.where(lidar_dists > threshold)[0]
#     ax.plot(angles_rad[valid_indices], lidar_dists[valid_indices], 'ro', 
#             label=f'Distance > {threshold}')
    
#     ax.set_title("LiDAR Scan in Polar Coordinates")
#     ax.legend(loc="upper right")
#     plt.show()

if __name__ == "__main__":
    main()
#     demo_find_angles()