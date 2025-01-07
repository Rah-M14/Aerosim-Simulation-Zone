import numpy as np
import random
from PIL import Image
from scipy.ndimage import binary_dilation

class RRTStarPlanner:
    def __init__(self, image_path, xlim=(-10, 10), ylim=(-7, 7), max_iter=5000, step_size=0.05, neighbor_radius=1.0):
        self.binary_map = self.convert_image_to_binary_map(image_path)
        self.xlim = xlim
        self.ylim = ylim
        self.max_iter = max_iter
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius

    def convert_image_to_binary_map(self, image_path, threshold=128, dilation_iterations=2):
        image = Image.open(image_path)
        gray_image = np.array(image.convert('L'))
        binary_map = (gray_image < threshold).astype(int)
        return binary_dilation(binary_map, iterations=dilation_iterations)

    def random_point_in_free_space(self):
        y_free, x_free = np.where(self.binary_map == 0)
        idx = random.randint(0, len(y_free) - 1)
        x = x_free[idx] / self.binary_map.shape[1] * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
        y = y_free[idx] / self.binary_map.shape[0] * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
        return x, y

    def compute_path(self, start_pos=None, goal_pos=None):
        # start = self.random_point_in_free_space()
        # goal = self.random_point_in_free_space()

        start_x, start_y = start_pos[0], start_pos[1]
        start = (start_x, start_y)
        goal_x, goal_y = goal_pos[0], goal_pos[1]
        goal = (goal_x, goal_y)

        rrt = RRTStar(start, goal, self.binary_map, self.xlim, self.ylim, 
                      self.max_iter, self.step_size, self.neighbor_radius)
        vertices, edges = rrt.run()
        
        return get_final_path(start, goal, edges), start, goal

    def plan(self, start_pos, goal_pos):
        path, start, goal = self.compute_path(start_pos, goal_pos)
        return {
            'path': path,
            'start': start,
            'goal': goal
        }

class RRTStar:
    def __init__(self, start, goal, binary_map, xlim, ylim, max_iter=5000, step_size=0.5, neighbor_radius=1.0):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.binary_map = binary_map
        self.xlim = xlim
        self.ylim = ylim
        self.max_iter = max_iter
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.vertices = [tuple(self.start)]
        self.edges = []
        self.costs = [0]

    def is_collision_free(self, point):
        x, y = point
        x_idx = int((x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.binary_map.shape[1])
        y_idx = int((y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.binary_map.shape[0])
        return self.binary_map[y_idx, x_idx] == 0

    def nearest_neighbor(self, point):
        distances = [np.linalg.norm(np.array(v) - point) for v in self.vertices]
        return np.array(self.vertices[np.argmin(distances)])

    def steer(self, from_point, to_point):
        direction = to_point - from_point
        distance = np.linalg.norm(direction)
        if distance > self.step_size:
            return tuple(from_point + direction / distance * self.step_size)
        return tuple(to_point)

    def near_neighbors(self, point):
        return [v for v in self.vertices if np.linalg.norm(np.array(v) - point) <= self.neighbor_radius]

    def run(self):
        for _ in range(self.max_iter):
            if random.random() < 0.05:
                sample = self.goal
            else:
                sample = self.random_point_in_free_space()

            nearest = self.nearest_neighbor(sample)
            new_point = self.steer(nearest, sample)

            if self.is_collision_free(new_point):
                near_vertices = self.near_neighbors(new_point)
                min_cost = float('inf')
                min_parent = None

                for near_vertex in near_vertices:
                    cost = self.costs[self.vertices.index(near_vertex)] + np.linalg.norm(np.array(near_vertex) - new_point)
                    if cost < min_cost:
                        min_cost = cost
                        min_parent = near_vertex

                self.vertices.append(new_point)
                self.edges.append((min_parent, new_point))
                self.costs.append(min_cost)

                # Rewire
                for near_vertex in near_vertices:
                    if not np.array_equal(near_vertex, min_parent):
                        cost = min_cost + np.linalg.norm(np.array(near_vertex) - new_point)
                        if cost < self.costs[self.vertices.index(near_vertex)]:
                            old_parent = [e[0] for e in self.edges if e[1] == near_vertex][0]
                            self.edges.remove((old_parent, near_vertex))
                            self.edges.append((new_point, near_vertex))
                            self.costs[self.vertices.index(near_vertex)] = cost

            if np.linalg.norm(np.array(new_point) - self.goal) < self.step_size:
                self.vertices.append(tuple(self.goal))
                self.edges.append((new_point, tuple(self.goal)))
                break

        return self.vertices, self.edges

    def random_point_in_free_space(self):
        while True:
            x = random.uniform(self.xlim[0], self.xlim[1])
            y = random.uniform(self.ylim[0], self.ylim[1])
            if self.is_collision_free((x, y)):
                return x, y

def get_final_path(start, goal, edges):
    path = []
    current = tuple(goal)
    while current != tuple(start):
        next_pos = current
        for edge in reversed(edges):
            if edge[1] == current:
                current = edge[0]
                break
        path.append(np.array(next_pos))
    path.append(np.array(start))
    return list(reversed(path))

if __name__ == "__main__":
    image_path = "standalone_examples/api/omni.isaac.kit/WR_World.png"
    planner = RRTStarPlanner(image_path)
    result = planner.plan(np.array([-7.0,-8.0]), np.array([-8.5, -9.5]))
    
    print(f"Start: ({result['start'][0]:.2f}, {result['start'][1]:.2f})")
    print(f"Goal: ({result['goal'][0]:.2f}, {result['goal'][1]:.2f})")
    print(type(result['path']))
    print(result['path'][0])
    print(type(result['path'][0]))

    print("Path:")
    # for point_pair in result['path']:
        # print(point_pair)
    
    print(result['path'])