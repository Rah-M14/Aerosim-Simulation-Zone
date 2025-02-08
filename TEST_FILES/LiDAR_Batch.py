import numpy as np
import cv2

def get_batched_lidar_points(binary_img, current_pos_batch, world_limits, num_rays=360, max_range=4.0):
    """
    Get LiDAR first contact points for a batch of sensor positions.
    
    Args:
        binary_img (np.ndarray): Binary image (2D array) where 0 indicates an obstacle and 1 free space.
        current_pos_batch (np.ndarray): Array of shape (B,2) containing the sensor's (x,y) world coordinates for each sample.
        world_limits (np.ndarray): Array of shape (2,2) specifying [[min_x, max_x], [min_y, max_y]] of the world.
        num_rays (int): Number of rays to cast (default 360 for 1° resolution).
        max_range (float): Maximum sensor range in world units.
        
    Returns:
        contact_points (np.ndarray): Array of shape (B, num_rays, 2) containing the (x,y) relative contact points for each ray.
            For rays that do not hit an obstacle within max_range, the contact point remains [0,0].
        lidar_dists (np.ndarray): Array of shape (B, num_rays) containing the Euclidean distance (in world units) to the obstacle.
            If no obstacle is detected within max_range, the distance is np.inf.
    """
    # Image and world scaling.
    height, width = binary_img.shape
    world_width = world_limits[0][1] - world_limits[0][0]
    world_height = world_limits[1][1] - world_limits[1][0]
    scale_x = width / world_width
    scale_y = height / world_height

    # Batch size.
    B = current_pos_batch.shape[0]

    # Convert world current positions to image coordinates.
    # (Note: using rint to keep proper centering.)
    sensor_x = np.rint((current_pos_batch[:, 0] - world_limits[0][0]) * scale_x).astype(np.int32)
    sensor_y = height - np.rint((current_pos_batch[:, 1] - world_limits[1][0]) * scale_y).astype(np.int32)
    sensor_pos_img = np.stack([sensor_x, sensor_y], axis=1)  # shape: (B, 2)

    # Convert max_range from world units to pixels. We use the minimum scaling factor.
    max_range_px = int(max_range * min(scale_x, scale_y))
    N = max_range_px - 1  # number of steps along each ray

    # Generate all angles and corresponding direction vectors (for image coordinates).
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)   # shape: (num_rays,)
    # In our image coordinates, x increases to the right and y increases downward.
    # The original function uses: [cos(angle), -sin(angle)]
    directions = np.stack([np.cos(angles), -np.sin(angles)], axis=0)  # shape: (2, num_rays)
    # Transpose to get shape (num_rays, 2)
    directions_T = directions.T

    # Generate distances from 1 to max_range_px - 1 along the rays.
    ray_lengths = np.arange(1, max_range_px)   # shape: (N,)

    # Compute the ray displacements for a single sensor position.
    # base_ray_points will have shape (N, num_rays, 2): for each step along the ray and for each ray.
    base_ray_points = ray_lengths[:, None, None] * directions_T[None, :, :]

    # Now, for each sensor, add its image coordinate.
    # Expand sensor_pos_img from (B, 2) to (B, 1, 1, 2) so that it broadcasts with base_ray_points.
    sensor_pos_img_expanded = sensor_pos_img[:, None, None, :]  # shape: (B, 1, 1, 2)
    # ray_points for all samples: shape becomes (B, N, num_rays, 2)
    ray_points = sensor_pos_img_expanded + base_ray_points[None, :, :, :]
    ray_points = ray_points.astype(np.int32)

    # Create a validity mask for points that lie within the image boundaries.
    valid_x = (ray_points[..., 0] >= 0) & (ray_points[..., 0] < width)
    valid_y = (ray_points[..., 1] >= 0) & (ray_points[..., 1] < height)
    valid_mask = valid_x & valid_y  # shape: (B, N, num_rays)

    # Initialize output arrays.
    contact_points = np.zeros((B, num_rays, 2), dtype=np.float32)
    lidar_dists = np.ones((B, num_rays), dtype=np.float32) * np.inf

    # Process each sample in the batch.
    for b in range(B):
        # For each ray in the sample.
        for r in range(num_rays):
            valid_indices = np.where(valid_mask[b, :, r])[0]
            if valid_indices.size == 0:
                continue

            # Get the pixel coordinates along the ray.
            coords = ray_points[b, valid_indices, r]  # shape: (k, 2)
            # Retrieve pixel values from binary_img (note: image indexing is (row=y, col=x)).
            ray_values = binary_img[coords[:, 1], coords[:, 0]]
            obstacle_idx = np.where(ray_values == 0)[0]
            if obstacle_idx.size > 0:
                first_step = valid_indices[obstacle_idx[0]]
                # Pixel coordinates of the first obstacle.
                px = ray_points[b, first_step, r, 0]
                py = ray_points[b, first_step, r, 1]
                # Convert back to world coordinates.
                # Adding 0.5 to convert from pixel index to pixel center.
                world_x = ((px + 0.5) / scale_x) + world_limits[0][0]
                world_y = world_limits[1][0] + ((height - (py + 0.5)) / scale_y)
                # Relative coordinates from sensor position.
                rel_x = world_x - current_pos_batch[b, 0]
                rel_y = world_y - current_pos_batch[b, 1]
                # Compute Euclidean distance.
                dist = np.sqrt(rel_x**2 + rel_y**2)
                if dist <= max_range:
                    contact_points[b, r, :] = [rel_x, rel_y]
                    lidar_dists[b, r] = dist

    return contact_points, lidar_dists


import matplotlib.pyplot as plt
import time

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
        contacts, dists = get_batched_lidar_points(binary_img, sensors, world_limits,
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

if __name__ == "__main__":
    main()