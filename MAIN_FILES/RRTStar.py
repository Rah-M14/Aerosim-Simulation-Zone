import numpy as np
import random
from PIL import Image
from scipy.ndimage import binary_dilation
from dataclasses import dataclass
from typing import Optional, List, Tuple
import cv2

from configs import RRTStarConfig

rrt_config = RRTStarConfig()

@dataclass
class Node:
    position: np.ndarray
    cost: float
    parent: Optional['Node'] = None

class RRTStarPlanner:
    def __init__(self, image_path, xlim=(-10, 10), ylim=(-7, 7), max_iter=rrt_config.max_iter, step_size=rrt_config.step_size, neighbor_radius=rrt_config.neighbor_radius):
        self.binary_map = self.convert_image_to_binary_map(image_path)
        self.xlim = xlim
        self.ylim = ylim
        self.max_iter = max_iter
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius

    def convert_image_to_binary_map(self, image_path, threshold=150, dilation_iterations=2):
        image = Image.open(image_path)
        gray_image = np.array(image.convert('L'))
        binary_map = (gray_image < threshold).astype(int)
        return binary_dilation(binary_map, iterations=dilation_iterations)

    def random_point_in_free_space(self):
        y_free, x_free = np.where(self.binary_map == 0)
        idx = random.randint(0, len(y_free) - 1)
        x = x_free[idx] / self.binary_map.shape[1] * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
        y = y_free[idx] / self.binary_map.shape[0] * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
        return np.array([x, y])

    def compute_path(self, start_pos=None, goal_pos=None):
        start = np.array(start_pos[:2])  # Only take x, y coordinates
        goal = np.array(goal_pos[:2])

        rrt = BiRRTStar(start, goal, self.binary_map, self.xlim, self.ylim, 
                        self.max_iter, self.step_size, self.neighbor_radius)
        path = rrt.run()
        
        return path, tuple(start), tuple(goal)

    def plan(self, start_pos, goal_pos):
        path, start, goal = self.compute_path(start_pos, goal_pos)
        return {
            'path': path,
            'start': start,
            'goal': goal
        }

class BiRRTStar:
    def __init__(self, start, goal, binary_map, xlim, ylim, max_iter=500, step_size=0.5, neighbor_radius=1.0):
        self.start = start
        self.goal = goal
        self.binary_map = binary_map
        self.xlim = xlim
        self.ylim = ylim
        self.max_iter = max_iter
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        
        # Initialize both trees
        self.start_tree = [Node(position=start, cost=0)]
        self.goal_tree = [Node(position=goal, cost=0)]

    def is_collision_free(self, point):
        x, y = point
        x_idx = int((x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.binary_map.shape[1])
        y_idx = int((y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.binary_map.shape[0])
        
        # Check bounds
        if not (0 <= x_idx < self.binary_map.shape[1] and 0 <= y_idx < self.binary_map.shape[0]):
            return False
            
        return self.binary_map[y_idx, x_idx] == 0

    def nearest_neighbor(self, point, tree):
        distances = [np.linalg.norm(node.position - point) for node in tree]
        return tree[np.argmin(distances)]

    def steer(self, from_point, to_point):
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        if distance > self.step_size:
            return from_point + direction / distance * self.step_size
        return to_point

    def near_neighbors(self, point, tree):
        return [node for node in tree 
                if np.linalg.norm(node.position - point) <= self.neighbor_radius]

    def extend_tree(self, tree, point):
        nearest = self.nearest_neighbor(point, tree)
        new_point = self.steer(nearest.position, point)
        
        if not self.is_collision_free(new_point):
            return None

        # Find best parent
        near_nodes = self.near_neighbors(new_point, tree)
        min_cost = nearest.cost + np.linalg.norm(nearest.position - new_point)
        best_parent = nearest

        for near_node in near_nodes:
            cost = near_node.cost + np.linalg.norm(near_node.position - new_point)
            if cost < min_cost and self.is_collision_free(new_point):
                min_cost = cost
                best_parent = near_node

        new_node = Node(position=new_point, cost=min_cost, parent=best_parent)
        tree.append(new_node)

        # Rewire
        for near_node in near_nodes:
            potential_cost = new_node.cost + np.linalg.norm(new_node.position - near_node.position)
            if potential_cost < near_node.cost and self.is_collision_free(near_node.position):
                near_node.parent = new_node
                near_node.cost = potential_cost

        return new_node

    def try_connect(self, node, tree):
        current = node
        while True:
            nearest = self.nearest_neighbor(current.position, tree)
            new_point = self.steer(current.position, nearest.position)
            
            if not self.is_collision_free(new_point):
                return None
                
            if np.array_equal(new_point, nearest.position):
                return nearest
                
            current = Node(position=new_point, cost=0)

    def get_path(self, start_node, goal_node):
        path = []
        
        # Get path from start node
        current = start_node
        while current:
            path.append(current.position)
            current = current.parent
            
        # Get path from goal node
        current = goal_node
        reverse_path = []
        while current:
            reverse_path.append(current.position)
            current = current.parent
            
        # Combine paths
        return np.array(path[::-1] + reverse_path)

    def random_point_in_free_space(self):
        y_free, x_free = np.where(self.binary_map == 0)
        idx = random.randint(0, len(y_free) - 1)
        x = x_free[idx] / self.binary_map.shape[1] * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
        y = y_free[idx] / self.binary_map.shape[0] * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
        return np.array([x, y])

    def run(self):
        for i in range(self.max_iter):
            # Sample random point with goal bias
            if random.random() < 0.05:
                random_point = self.goal
            else:
                random_point = self.random_point_in_free_space()

            # Extend start tree
            new_node = self.extend_tree(self.start_tree, random_point)
            if new_node:
                # Try to connect to goal tree
                goal_connection = self.try_connect(new_node, self.goal_tree)
                if goal_connection:
                    return self.get_path(new_node, goal_connection)

            # Extend goal tree
            new_node = self.extend_tree(self.goal_tree, random_point)
            if new_node:
                # Try to connect to start tree
                start_connection = self.try_connect(new_node, self.start_tree)
                if start_connection:
                    return self.get_path(start_connection, new_node)

        return []  # Return empty path if no solution found

def gen_goal_pose():
    new_pos = np.array(
        [
            np.random.choice(
                list(
                    set([x for x in np.linspace(-7.5, 7.6, 10000)])
                    - set(
                        y
                        for y in np.append(
                            np.linspace(-2.6, -1.7, 900),
                            np.append(
                                np.linspace(-0.8, 0.4, 1200),
                                np.append(
                                    np.linspace(1.5, 2.4, 900),
                                    np.linspace(3.4, 4.6, 1200),
                                ),
                            ),
                        )
                    )
                )
            ),
            np.random.choice(
                list(
                    set([x for x in np.linspace(-5.5, 5.6, 14000)])
                    - set(
                        y
                        for y in np.append(
                            np.linspace(-1.5, 2.5, 1000),
                            np.linspace(-2.5, -5.6, 3100),
                        )
                    )
                )
            ),
            0.0,
        ]
    )
    return new_pos

def visualize_path(image_path, start, goal, path, test_number, binary_map):
    # Load the original image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (1090, 720))  # Half the size for side-by-side display
    
    # Prepare binary map for visualization
    binary_vis = (binary_map * 255).astype(np.uint8)  # Convert to 0-255 range
    binary_vis = cv2.resize(binary_vis, (1090, 720))
    # Convert to 3-channel for colored path drawing
    binary_vis = cv2.cvtColor(binary_vis, cv2.COLOR_GRAY2BGR)
    
    # Convert world coordinates to image coordinates
    height, width = img.shape[:2]
    xlim = (-10, 10)
    ylim = (-7, 7)
    
    def world_to_image(point):
        x = int((point[0] - xlim[0]) / (xlim[1] - xlim[0]) * width)
        y = int((point[1] - ylim[0]) / (ylim[1] - ylim[0]) * height)
        return x, y
    
    # Create copies for drawing
    img_with_path = img.copy()
    binary_with_path = binary_vis.copy()
    
    # Plot start and goal points on both images
    start_x, start_y = world_to_image(start)
    goal_x, goal_y = world_to_image(goal)
    
    # Draw start point (green) and goal point (red) on both images
    for canvas in [img_with_path, binary_with_path]:
        cv2.circle(canvas, (start_x, start_y), 5, (0, 255, 0), -1)  # Green
        cv2.circle(canvas, (goal_x, goal_y), 5, (0, 0, 255), -1)   # Red
    
    # Plot path on both images
    if len(path) > 0:
        path_points = np.array([world_to_image(point) for point in path], dtype=np.int32)
        for i in range(len(path_points) - 1):
            pt1 = tuple(path_points[i])
            pt2 = tuple(path_points[i + 1])
            for canvas in [img_with_path, binary_with_path]:
                cv2.line(canvas, pt1, pt2, (255, 0, 0), 2)  # Blue
    
    # Combine images side by side
    combined_img = np.hstack((img_with_path, binary_with_path))
    
    # Add text for test number and path length
    text = f"Test {test_number} - Path Length: {len(path)} points"
    cv2.putText(combined_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 0, 0), 2)
    
    # Add labels for each view
    cv2.putText(combined_img, "Original Map", (10, 350), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(combined_img, "Binary Map", (550, 350), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Show the combined image
    cv2.imshow('RRT* Path Planning', combined_img)
    cv2.waitKey(100)  # Wait for 100ms

if __name__ == "__main__":
    image_path = "standalone_examples/api/omni.isaac.kit/MAIN_FILES/New_WR_World.png"
    planner = RRTStarPlanner(image_path)

    try:
        for i in range(100):
            print(f"\nTest {i+1}/100")
            start_pos = gen_goal_pose()
            goal_pos = gen_goal_pose()
            result = planner.plan(start_pos, goal_pos)
            
            print(f"Start: ({result['start'][0]:.2f}, {result['start'][1]:.2f})")
            print(f"Goal: ({result['goal'][0]:.2f}, {result['goal'][1]:.2f})")
            print(f"Path points: {len(result['path'])}")
            
            # Visualize the path with binary map
            visualize_path(
                image_path,
                result['start'],
                result['goal'],
                result['path'],
                i+1,
                planner.binary_map  # Pass the binary map to visualization
            )
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopping visualization...")
    
    finally:
        # Cleanup
        cv2.destroyAllWindows()