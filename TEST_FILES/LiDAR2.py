import numpy as np
import cv2

def get_lidar_points(binary_img, current_pos, world_limits, num_rays=360, max_range=4.0):
    """
    Get LiDAR first contact points in world coordinates
    Args:
        binary_img: Binary image where 0 is obstacle, 1 is free space
        current_pos: (x,y) position of the sensor in world coordinates
        world_limits: Array of [[min_x, max_x], [min_y, max_y]] world boundaries
        num_rays: Number of rays to cast (default 360 for 1-degree resolution)
        max_range: Maximum range of the sensor in world units
    Returns:
        points: Array of (x,y) coordinates relative to sensor position
    """
    height, width = binary_img.shape
    points = []
    
    # Calculate transformation factors from world to image
    world_width = world_limits[0][1] - world_limits[0][0]   # max_x - min_x
    world_height = world_limits[1][1] - world_limits[1][0]  # max_y - min_y
    
    # Calculate scales with proper orientation
    scale_x = width / world_width
    scale_y = height / world_height
    
    # Convert world position to image coordinates
    # Adjust y-coordinate transformation to match world coordinates exactly
    img_x = int((current_pos[0] - world_limits[0][0]) * scale_x)
    img_y = height - int((current_pos[1] - world_limits[1][0]) * scale_y)  # Flip y-axis and adjust origin
    
    # Convert max_range to pixels
    max_range_px = int(max_range * min(scale_x, scale_y))
    
    # Cast rays in all directions
    angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
    for angle in angles:
        # Calculate direction vector
        dx = np.cos(angle)
        dy = -np.sin(angle)  # Invert y component for image space
        
        found_obstacle = False
        # Ray march until hit or max range
        for r in range(1, max_range_px):
            # Calculate current point in image coordinates
            px = int(img_x + r * dx)
            py = int(img_y + r * dy)
            
            # Check bounds
            if px < 0 or px >= width or py < 0 or py >= height:
                break
                
            # Check if hit obstacle (black pixel)
            if binary_img[py, px] == 0:
                # Convert back to world coordinates
                world_x = (px / scale_x) + world_limits[0][0]
                world_y = world_limits[1][0] + (height - py) / scale_y  # Correct y-axis transformation
                
                # Store point relative to sensor position
                rel_x = world_x - current_pos[0]
                rel_y = world_y - current_pos[1]
                
                # Only add point if within max range
                if np.sqrt(rel_x**2 + rel_y**2) <= max_range:
                    points.append([rel_x, rel_y])
                found_obstacle = True
                break
    
    return np.array(points) if points else np.zeros((0, 2))