import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def lidar_simulation_from_image(image_path, left_corner, right_corner, start, end, resolution, fov=360, num_rays=720, max_distance=250):
    img = np.array(Image.open(image_path).convert('L'))  # Convert to grayscale
    binary_img = (img > 128).astype(np.uint8)  # Threshold to create a binary map
    binary_img = cv2.resize(binary_img, (0,0), fx=0.25, fy=0.25)
    img_height, img_width = binary_img.shape
    resolution_x = img_width / (right_corner[0] - left_corner[0])
    resolution_y = img_height / (right_corner[1] - left_corner[1])
    distances = []
    lidar_image = np.ones_like(binary_img) * 255  # Initialize a white background for LiDAR image
    
    def to_image_coords(coord):
        x, y = coord
        img_x = int((x - left_corner[0]) * resolution_x)
        img_y = int((y - left_corner[1]) * resolution_y)  # Correct y-axis scaling and no flipping
        return img_x, img_y
    
    start_x, start_y = to_image_coords(start)  # LiDAR start
    
    # lidar_image_color = cv2.cvtColor(lidar_image, cv2.COLOR_GRAY2BGR)
    # Simulate LiDAR beams
    angles = np.linspace(0, np.deg2rad(fov), num_rays)
    # Vectorized ray calculations
    for angle in angles:
        dx, dy = np.cos(angle), -np.sin(angle)  # Negate dy to match image coordinates
        distance = 0
        x, y = start_x, start_y
        max_distance = max_distance
        # Move along the ray and check for objects until we exit the image bounds
        while 0 <= x < img_width and 0 <= y < img_height and distance < max_distance:
            if binary_img[int(y), int(x)] == 0:  # Check for an object
                distances.append((distance, angle))
                break  # Found an object, stop the ray
            x += dx * resolution
            y += dy * resolution
            distance += resolution
    
    # Convert polar coordinates to Cartesian for visualization
    points = [(d * np.cos(a), d * np.sin(a)) for d, a in distances]
    points = np.array(points)
    # Plot the original binary image and LiDAR scan
    # fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # # Plot the simulated LiDAR scan (in binary image)
    # axes[0].imshow(binary_img, cmap='gray', extent=(0, img_width, img_height, 0))
    # axes[0].scatter(start_x, start_y, color='red', label='LiDAR Position', s=50)
    # axes[0].scatter(points[:, 0] / resolution + start_x, points[:, 1] / resolution + start_y, s=2, color='blue', label='LiDAR Points')
    # axes[0].legend()
    # axes[0].set_title("Simulated LiDAR Scan")
    # axes[0].set_xlabel("X")
    # axes[0].set_ylabel("Y")
    # axes[0].grid(True)
    # # Draw the LiDAR points in the lidar_image using OpenCV
    # for point in points:
    #     lidar_x = int(point[0] / resolution + start_x)
    #     lidar_y = int(point[1] / resolution + start_y)
    #     cv2.circle(lidar_image_color, (lidar_x, lidar_y), 5, (255, 0, 0), -1)  # Blue circles for LiDAR points
    # # Draw the start and end points
    # # cv2.circle(lidar_image_color, (start_x, start_y), 10, (0, 0, 255), -1)  # Red circle for start
    # # cv2.circle(lidar_image_color, (end_x, end_y), 10, (0, 255, 0), -1)  # Green circle for end
    # # lidar_image_color = cv2.resize(lidar_image_color, (256, 256))
    # # Plot the LiDAR-simulated image with drawn points
    # axes[1].imshow(cv2.cvtColor(lidar_image_color, cv2.COLOR_BGR2RGB), extent=(left_corner[0], right_corner[0], left_corner[1], right_corner[1]))
    # axes[1].set_title("LiDAR-Simulated Image with Points")
    # axes[1].set_xlabel("X")
    # axes[1].set_ylabel("Y")
    # axes[1].grid(True)
    # axes[1].legend()
    # plt.tight_layout()
    # plt.show()
    return points

if __name__ == "__main__":
    image_path = "/home/rahm/.local/share/ov/pkg/isaac-sim-4.2.0/standalone_examples/api/omni.isaac.kit/TEST_FILES/New_WR_World.png"  # Replace with the path to your image
    resolution = 1  # Distance step in pixels
    left = (-10, 7)
    right = (10, -7)
    start = (0, -3.5)
    end = (5, -1)
    fov = 360
    num_rays = 720
    max_distance = 250
    lidar_simulation_from_image(image_path, left, right, start, end, resolution, fov, num_rays, max_distance)