import numpy as np
import cv2

def get_lidar_points(binary_img, current_pos, world_limits, num_rays=360, max_range=4.0):
    """
    Get LiDAR first contact points in world coordinates using vectorized operations
    Args:
        binary_img: Binary image where 0 is obstacle, 1 is free space
        current_pos: (x,y) position of the sensor in world coordinates
        world_limits: Array of [[min_x, max_x], [min_y, max_y]] world boundaries
        num_rays: Number of rays to cast (default 360 for 1-degree resolution)
        max_range: Maximum range of the sensor in world units
    Returns:
        points: Array of shape (360, 2) with (x,y) coordinates relative to sensor position,
                zeros for rays that don't hit anything
    """
    height, width = binary_img.shape
    
    # Calculate transformation factors from world to image
    world_width = world_limits[0][1] - world_limits[0][0]
    world_height = world_limits[1][1] - world_limits[1][0]
    scale_x = width / world_width
    scale_y = height / world_height
    
    # Convert world position to image coordinates - use rounding to keep proper centering.
    img_x = int(round((current_pos[0] - world_limits[0][0]) * scale_x))
    img_y = height - int(round((current_pos[1] - world_limits[1][0]) * scale_y))
    
    # Convert max_range to pixels
    max_range_px = int(max_range * min(scale_x, scale_y))
    
    # Generate all angles at once
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    
    # Generate direction vectors for all angles
    directions = np.stack([np.cos(angles), -np.sin(angles)], axis=0)  # Shape: (2, num_rays)
    
    # Generate all ray lengths at once
    ray_lengths = np.arange(1, max_range_px)  # Shape: (max_range_px-1,)
    
    # Calculate all possible points for all rays using broadcasting
    ray_points = ray_lengths[:, np.newaxis, np.newaxis] * directions[np.newaxis, :, :]
    ray_points = np.transpose(ray_points, (0, 2, 1))  # Reshape to (max_range_px-1, num_rays, 2)
    
    # Add sensor position to all points
    ray_points_x = ray_points[..., 0] + img_x  # (max_range_px-1, num_rays)
    ray_points_y = ray_points[..., 1] + img_y
    
    # Convert to integer coordinates
    ray_points_x = ray_points_x.astype(np.int32)
    ray_points_y = ray_points_y.astype(np.int32)
    
    # Create masks for valid points
    valid_x = (ray_points_x >= 0) & (ray_points_x < width)
    valid_y = (ray_points_y >= 0) & (ray_points_y < height)
    valid_points = valid_x & valid_y
    
    # Initialize arrays to store contact points and distances
    contact_points = np.zeros((num_rays, 2))
    lidar_dists = np.ones(num_rays) * np.inf
    
    # Find first contact point for each ray
    for ray_idx in range(num_rays):
        valid_ray_points = valid_points[:, ray_idx]
        if not np.any(valid_ray_points):
            continue
            
        ray_x = ray_points_x[valid_ray_points, ray_idx]
        ray_y = ray_points_y[valid_ray_points, ray_idx]
        
        # Check for obstacles along the ray
        ray_values = binary_img[ray_y, ray_x]
        obstacle_indices = np.where(ray_values == 0)[0]
        
        if len(obstacle_indices) > 0:
            # Get first contact point
            first_contact_idx = obstacle_indices[0]
            px = ray_x[first_contact_idx]
            py = ray_y[first_contact_idx]
            
            # Convert back to world coordinates.
            # Adding a 0.5 offset so we convert from pixel centers.
            world_x = ((px + 0.5) / scale_x) + world_limits[0][0]
            world_y = world_limits[1][0] + ((height - (py + 0.5)) / scale_y)
            
            # Calculate relative coordinates from the sensor's current position.
            rel_x = world_x - current_pos[0]
            rel_y = world_y - current_pos[1]
            dist = np.sqrt(rel_x**2 + rel_y**2)
            
            if dist <= max_range:
                contact_points[ray_idx] = [rel_x, rel_y]
                lidar_dists[ray_idx] = dist
    
    return contact_points, lidar_dists