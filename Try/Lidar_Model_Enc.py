import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Import the required functions from LiDAR_Fast.py.
# (Make sure Try/LiDAR_Fast.py is in your PYTHONPATH or same folder so that the module import works.)
from LiDAR_Fast import get_lidar_points, create_binary_image

def group_angle_ranges(angles, mask, min_len=1):
    """
    Vectorized grouping of consecutive angles where the mask is True.
    
    This function now supports batched inputs. If angles and mask are 2D arrays
    of shape (batch_size, num_rays), it returns a list of groups for each batch element.
    
    Args:
        angles (np.ndarray): Array of angles (in degrees). Can be 1D (num_rays,) or 2D (batch, num_rays).
        mask (np.ndarray): Boolean array of the same shape as angles.
        min_len (int): Minimum number of consecutive rays required to form a group.
        
    Returns:
        groups (list): If input is 1D, returns a list of tuples (start_angle, end_angle).
                       If input is 2D, returns a list of lists (one per batch element) of tuples.
    """
    # If inputs are batched (2D arrays), process each batch element separately.
    if angles.ndim == 2:
        all_groups = []
        for ang_row, mask_row in zip(angles, mask):
            indices = np.nonzero(mask_row)[0]
            if indices.size == 0:
                all_groups.append([])
            else:
                diff = np.diff(indices)
                boundaries = np.where(diff != 1)[0]
                group_starts = np.concatenate(([0], boundaries + 1))
                group_ends = np.concatenate((boundaries, [len(indices) - 1]))
                groups_row = [
                    (ang_row[indices[start]], ang_row[indices[end]])
                    for start, end in zip(group_starts, group_ends)
                    if (end - start) >= min_len
                ]
                all_groups.append(groups_row)
        return all_groups
    else:
        indices = np.nonzero(mask)[0]
        if indices.size == 0:
            return []
        diff = np.diff(indices)
        boundaries = np.where(diff != 1)[0]
        group_starts = np.concatenate(([0], boundaries + 1))
        group_ends = np.concatenate((boundaries, [len(indices) - 1]))
        groups = [
            (angles[indices[start]], angles[indices[end]])
            for start, end in zip(group_starts, group_ends)
            if (end - start) >= min_len
        ]
        return groups

def process_safe_angles(lidar_dists, bot_orientation, threshold, num_rays=360,
                        finite_min_len=12, infinite_min_len=6, n_centres=3, 
                        goal_orientation=None):
    """
    Process the LiDAR distances and group safe angles into finite and infinite safe groups,
    vectorizing operations where possible. Supports both single-sample (1D) and batched (2D)
    lidar distances. For batched inputs, bot_orientation is assumed identical across scans.
    
    The function returns four NumPy arrays (all in radians) with fixed shapes:
        infinite_centres: (batch_size, n_centres)
        finite_centres:   (batch_size, n_centres)
        infinite_bounds:  (batch_size, n_centres, 2)  -- each row is the (start, end) pair in radians
        finite_bounds:    (batch_size, n_centres, 2)
        
    If the number of groups found is less than required, the arrays are padded with the
    provided goal_orientation (an input, in radians). If goal_orientation is None, bot_orientation
    is used.
    
    Args:
        lidar_dists (np.ndarray): 1D (num_rays,) or 2D (batch_size, num_rays) array.
        bot_orientation (float): Bot's orientation in radians.
        threshold (float): Distance threshold.
        num_rays (int): Number of rays (default 360).
        finite_min_len (int): Minimum group length for finite safe groups.
        infinite_min_len (int): Minimum group length for infinite safe groups.
        n_centres (int): Number of candidate centre angles (and corresponding boundaries) to output per scan (default 3).
        goal_orientation (float, optional): Value (in radians) used for padding; if None, bot_orientation.
    
    Returns:
        A tuple of four NumPy arrays:
          (infinite_centres, finite_centres, infinite_bounds, finite_bounds)
        with shapes (batch_size, n_centres), (batch_size, n_centres),
        (batch_size, n_centres, 2), (batch_size, n_centres, 2), respectively.
    """
    if goal_orientation is None:
        goal_orientation = bot_orientation
    # Work in degrees for sorting and padding.
    orientation_deg = np.rad2deg(bot_orientation)
    pad_deg = np.rad2deg(goal_orientation)

    lidar_dists = np.asarray(lidar_dists)
    
    def process_single_scan(scan):
        # Ensure scan has exactly num_rays entries.
        if scan.shape[0] != num_rays:
            scan = scan[:num_rays]
        angles_deg = np.linspace(0, 360, num_rays, endpoint=False)
        
        finite_mask = (scan > threshold) & (~np.isinf(scan))
        infinite_mask = np.isinf(scan)
        
        finite_groups = group_angle_ranges(angles_deg, finite_mask, min_len=finite_min_len)
        infinite_groups = group_angle_ranges(angles_deg, infinite_mask, min_len=infinite_min_len)
        
        # For each group, compute the centre (as the mean of the start and end) and retain the bounds.
        def groups_to_centres_and_bounds(groups):
            centres = []
            bounds = []
            for rng in groups:
                centre = (rng[0] + rng[1]) / 2.0  # centre in degrees
                centres.append(centre)
                bounds.append(rng)               # (start, end) pair in degrees
            if centres:
                centres = np.array(centres)
                bounds = np.array(bounds)  # shape (n, 2)
            else:
                centres = np.array([])
                bounds = np.empty((0, 2))
            return centres, bounds
        
        finite_centres_deg, finite_bounds_deg = groups_to_centres_and_bounds(finite_groups)
        infinite_centres_deg, infinite_bounds_deg = groups_to_centres_and_bounds(infinite_groups)
        
        # Sort groups by closeness of the centre to bot's orientation (in degrees)
        if finite_centres_deg.size > 0:
            order = np.argsort(np.abs(finite_centres_deg - orientation_deg))
            finite_centres_deg = finite_centres_deg[order]
            finite_bounds_deg = finite_bounds_deg[order]
        if infinite_centres_deg.size > 0:
            order = np.argsort(np.abs(infinite_centres_deg - orientation_deg))
            infinite_centres_deg = infinite_centres_deg[order]
            infinite_bounds_deg = infinite_bounds_deg[order]
        
        # Truncate or pad the groups to fixed sizes.
        def fixed_size(arr, desired_size, pad_val):
            if arr.size >= desired_size:
                return arr[:desired_size]
            else:
                pad = np.full(desired_size - arr.size, pad_val)
                return np.concatenate([arr, pad])
        
        def fixed_size_bounds(arr, desired_size, pad_val):
            # arr shape expected to be (n, 2)
            if arr.shape[0] >= desired_size:
                return arr[:desired_size]
            else:
                pad = np.full((desired_size - arr.shape[0], 2), pad_val)
                return np.concatenate([arr, pad], axis=0)
        
        finite_centres_deg = fixed_size(finite_centres_deg, n_centres, pad_deg)
        finite_bounds_deg  = fixed_size_bounds(finite_bounds_deg, n_centres, pad_deg)
        infinite_centres_deg = fixed_size(infinite_centres_deg, n_centres, pad_deg)
        infinite_bounds_deg  = fixed_size_bounds(infinite_bounds_deg, n_centres, pad_deg)
        
        # Convert all results to radians.
        return (np.deg2rad(infinite_centres_deg),
                np.deg2rad(finite_centres_deg),
                np.deg2rad(infinite_bounds_deg),
                np.deg2rad(finite_bounds_deg))
    
    # Process as a batch.
    if lidar_dists.ndim == 1:
        ic, fc, ib, fb = process_single_scan(lidar_dists)
        # Expand dims to simulate a batch of size 1.
        ic = ic.reshape(1, -1)
        fc = fc.reshape(1, -1)
        ib = ib.reshape(1, -1, 2)
        fb = fb.reshape(1, -1, 2)
    elif lidar_dists.ndim == 2:
        batch_size = lidar_dists.shape[0]
        ic_list, fc_list, ib_list, fb_list = [], [], [], []
        for i in range(batch_size):
            ic_i, fc_i, ib_i, fb_i = process_single_scan(lidar_dists[i])
            ic_list.append(ic_i)
            fc_list.append(fc_i)
            ib_list.append(ib_i)
            fb_list.append(fb_i)
        ic = np.vstack(ic_list)
        fc = np.vstack(fc_list)
        ib = np.vstack(ib_list)
        fb = np.vstack(fb_list)
    else:
        raise ValueError("lidar_dists must be a 1D or 2D array.")
    
    return ic, fc, ib, fb

def plot_safe_angles_on_image(binary_img, sensor_pos, world_limits, threshold, bot_orientation, num_rays=360, max_range=4.0):
    """
    Vectorized LiDAR scan plotting and processing: plots safe angles on the binary image,
    computes safe angle groups (both finite and infinite), sorts them in decreasing order
    of deviation from bot_orientation, and returns group details.

    Args:
        binary_img (np.ndarray): Binary image (2D) used for the LiDAR map.
        sensor_pos (np.ndarray): Sensor's (x,y) world coordinates.
        world_limits (np.ndarray): World boundaries as [[min_x, max_x], [min_y, max_y]].
        threshold (float): Distance threshold.
        bot_orientation (float): The bot's orientation in degrees.
        num_rays (int): Number of LiDAR rays to cast (default 360).
        max_range (float): Maximum LiDAR range in world units.
        
    Returns:
        infinite_centres (list): Centre angles for infinite safe groups.
        finite_centres (list): Centre angles for finite safe groups.
        infinite_ranges (list): Angle ranges for infinite safe groups.
        finite_ranges (list): Angle ranges for finite safe groups.
    """
    # Run the LiDAR scan.
    contact_points, lidar_dists = get_lidar_points(binary_img, sensor_pos, world_limits,
                                                   num_rays=num_rays, max_range=max_range)
    
    angles_deg = np.linspace(0, 360, num_rays, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)
    
    safe_mask = (lidar_dists > threshold)
    infinite_mask = np.isinf(lidar_dists)
    finite_mask = safe_mask & (~np.isinf(lidar_dists))
    
    # Process safe angles in a vectorized way.
    infinite_centres, finite_centres, infinite_bounds, finite_bounds = process_safe_angles(
        lidar_dists, bot_orientation, threshold, num_rays=num_rays, finite_min_len=6, infinite_min_len=6)
    
    # Prepare conversion parameters.
    height, width = binary_img.shape
    world_width = world_limits[0][1] - world_limits[0][0]
    world_height = world_limits[1][1] - world_limits[1][0]
    scale_x = width / world_width
    scale_y = height / world_height
    
    sensor_img_x = int(round((sensor_pos[0] - world_limits[0][0]) * scale_x))
    sensor_img_y = height - int(round((sensor_pos[1] - world_limits[1][0]) * scale_y))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(binary_img, cmap='gray', origin='upper')
    ax.plot(sensor_img_x, sensor_img_y, 'go', markersize=10, label='Sensor')
    
    # Plot finite safe rays (vectorized using LineCollection).
    finite_indices = np.nonzero(finite_mask)[0]
    if finite_indices.size:
        endpoints = sensor_pos + contact_points[finite_indices, :]
        ep_img_x = (endpoints[:, 0] - world_limits[0][0]) * scale_x
        ep_img_y = height - ((endpoints[:, 1] - world_limits[1][0]) * scale_y)
        sensor_coords = np.array([sensor_img_x, sensor_img_y])
        finite_segments = np.stack([np.tile(sensor_coords, (endpoints.shape[0], 1)),
                                      np.column_stack((ep_img_x, ep_img_y))], axis=1)
        lc_finite = LineCollection(finite_segments, colors='r', linewidths=1)
        ax.add_collection(lc_finite)
        ax.scatter(ep_img_x, ep_img_y, c='r', s=9)
    
    # Plot infinite safe rays (vectorized).
    infinite_indices = np.nonzero(infinite_mask)[0]
    if infinite_indices.size:
        relevant_angles = angles_rad[infinite_indices]
        endpoints = sensor_pos + max_range * np.column_stack((np.cos(relevant_angles), np.sin(relevant_angles)))
        ep_img_x = (endpoints[:, 0] - world_limits[0][0]) * scale_x
        ep_img_y = height - ((endpoints[:, 1] - world_limits[1][0]) * scale_y)
        sensor_coords = np.array([sensor_img_x, sensor_img_y])
        infinite_segments = np.stack([np.tile(sensor_coords, (endpoints.shape[0], 1)),
                                        np.column_stack((ep_img_x, ep_img_y))], axis=1)
        lc_infinite = LineCollection(infinite_segments, colors='b', linestyles='dashed', linewidths=1)
        ax.add_collection(lc_infinite)
        ax.scatter(ep_img_x, ep_img_y, c='b', s=9)
    
    # Plot centre markers for the groups.
    if finite_centres.size > 0:
        # Flatten the centres array so that it is 1D.
        finite_centres_arr = np.deg2rad(np.array(finite_centres).flatten())
        endpoints = sensor_pos + max_range * 0.5 * np.column_stack((np.cos(finite_centres_arr), np.sin(finite_centres_arr)))
        ep_img_x = (endpoints[:, 0] - world_limits[0][0]) * scale_x
        ep_img_y = height - ((endpoints[:, 1] - world_limits[1][0]) * scale_y)
        ax.scatter(ep_img_x, ep_img_y, c='k', marker='*', s=100)
    
    if infinite_centres.size > 0:
        infinite_centres_arr = np.deg2rad(np.array(infinite_centres).flatten())
        endpoints = sensor_pos + max_range * 0.5 * np.column_stack((np.cos(infinite_centres_arr), np.sin(infinite_centres_arr)))
        ep_img_x = (endpoints[:, 0] - world_limits[0][0]) * scale_x
        ep_img_y = height - ((endpoints[:, 1] - world_limits[1][0]) * scale_y)
        ax.scatter(ep_img_x, ep_img_y, c='c', marker='*', s=100)
    
    ax.set_title("LiDAR Safe Angles (Finite: red solid, Infinite: blue dashed)")
    ax.legend()
    plt.show()
    
    return infinite_centres, finite_centres, infinite_bounds, finite_bounds

def demo_plot_safe_angles():
    """
    Demo function to showcase the fast, vectorized processing and plotting of LiDAR safe angles.
    Adjust image source, world limits, sensor position, threshold, and bot_orientation as needed.
    """
    # Load a binary image (adjust path as needed). Alternatively, use create_synthetic_binary_image().
    binary_img = create_binary_image(image_path=r"F:\Aerosim-Simulation-Zone\Try\New_WR_World.png")
    
    # Define world limits.
    world_limits = np.array([[-10, 10], [-8, 8]])
    
    # Set the sensor position.
    sensor_pos = np.array([-5, -0.5])
    
    # Define the distance threshold.
    threshold = 3.0
    
    # Bot's orientation.
    bot_orientation = 0.0
    
    # Execute the vectorized plotting and processing.
    infinite_centres, finite_centres, infinite_bounds, finite_bounds = plot_safe_angles_on_image(
        binary_img, sensor_pos, world_limits, threshold, bot_orientation, num_rays=360, max_range=4.0)
    
    print("Returned Infinite Centre Angles:", infinite_centres)
    print("Returned Finite Centre Angles:", finite_centres)
    print("Returned Infinite Angle Ranges:", infinite_bounds)
    print("Returned Finite Angle Ranges:", finite_bounds)

if __name__ == "__main__":
    demo_plot_safe_angles()
