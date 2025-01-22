import numpy as np
from RRTStar import RRTStarPlanner

class PathManager:
    def __init__(self, image_path, chunk_size=10, max_attempts=5):
        self.planner = RRTStarPlanner(image_path)
        self.chunk_size = chunk_size
        self.max_attempts = max_attempts
        # self.overlap_size = int(chunk_size * overlap_ratio)
        self.current_path = None
        self.current_index = 0
        self.new_chunk = None
        self.start = None
        self.goal = None
        self.bot_length = 1.0
        
    def plan_new_path(self, start_pos, goal_pos):
        for attempt in range(self.max_attempts):
            self.start = start_pos
            self.goal = goal_pos
            
            result = self.planner.plan(start_pos, goal_pos)
            self.current_path = result['path']

            if self.current_path is not None and len(self.current_path) > 0:
                # check if current path is a numpy atray if not, make it one
                if not isinstance(self.current_path, np.ndarray):
                    self.current_path = np.array(self.current_path)
                self.current_path = self.interpolate_path(self.current_path, self.bot_length)
                self.current_index = 0
                self.new_chunk = np.array(self.current_path[self.current_index:self.current_index + self.chunk_size])
                return True
                # return result            
            
        print(f"Warning: Failed to generate path after {self.max_attempts} attempts")
        return False
    
    def get_next_chunk(self, current_pos):
        if self.current_path is None or len(self.current_path) == 0:
            # Return a default chunk if no path exists
            return np.tile(current_pos, (self.chunk_size, 1))

        # Update current_index if close to current waypoint
        if len(self.new_chunk) > 0 and np.linalg.norm(current_pos - self.new_chunk[0]) < 0.2:
            self.current_index += 1
        elif np.linalg.norm(current_pos - self.new_chunk[1]) < 0.2:
            self.current_index += 2

        # Get remaining path from current index
        remaining_path = self.current_path[self.current_index:]
        
        if len(remaining_path) == 0:
            # If no remaining path, return last position repeated
            return np.tile(current_pos, (self.chunk_size, 1))
        
        # Take next chunk_size points or pad with last point if needed
        if len(remaining_path) >= self.chunk_size:
            self.new_chunk = remaining_path[:self.chunk_size]
        else:
            # Pad with the last waypoint if path is shorter than chunk_size
            last_point = remaining_path[-1]
            padding_size = self.chunk_size - len(remaining_path)
            padding = np.tile(last_point, (padding_size, 1))
            self.new_chunk = np.vstack([remaining_path, padding])
        
        assert self.new_chunk.shape == (self.chunk_size, 2), f"Incorrect chunk shape: {self.new_chunk.shape}"
        return self.new_chunk
    
    # def get_chunk_progress(self, current_pos):
    #     print(f"Current position in chunk progress: {current_pos}")
    #     if np.linalg.norm(current_pos - self.new_chunk[0]) < 0.01:
    #         return 0
        
    #     min_dist = float('inf')
    #     closest_idx = 0
    #     for i, coords in enumerate(self.new_chunk):
    #         dist = np.linalg.norm(current_pos - coords)
    #         if dist < min_dist:
    #             min_dist = dist
    #             closest_idx = i
    #     return closest_idx

    # def needs_new_chunk(self, current_pos):
    #     if self.new_chunk is None:
    #         return True
        
    #     progress_idx = self.get_chunk_progress(current_pos)
    #     return progress_idx >= self.overlap_size and self.current_index < len(self.current_path)
    
    def reset(self):
        self.current_index = 0
        self.new_chunk = None
        self.current_path = None
        self.start = None
        self.goal = None

    def get_full_path(self):
        """Returns the full path if it exists, otherwise an empty list."""
        return self.current_path if isinstance(self.current_path, list) or (isinstance(self.current_path, np.ndarray) and self.current_path.size > 0) else []

    def interpolate_path(self, path, bot_length):
        interpolated_path = []
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            in_dist = np.linalg.norm(end - start)
            if in_dist > bot_length:
                intermediate_points = np.linspace(start, end, num=int(in_dist // bot_length), endpoint=True)
            else:
                intermediate_points = np.array([start, end])
            interpolated_path.extend(intermediate_points)
        return np.array(interpolated_path)
    
if __name__ == "__main__":
    image_path = "standalone_examples/api/omni.isaac.kit/WR_World.png"
    path_manager = PathManager(image_path, chunk_size=10)
    
    start_pos = np.array([-4.0, -3.0])
    goal_pos = np.array([6.0, 5.0])
    result = path_manager.plan_new_path(start_pos, goal_pos)
    
    current_pos = start_pos
    while True:
        # idx = path_manager.get_chunk_progress(current_pos)
        chunk = path_manager.get_next_chunk(current_pos)
        
        if not (chunk[0]==current_pos).all():
            print("\nNew chunk loaded!")
        
        print(f"\nCurrent position: {current_pos}")
        print(f"Progress through chunk: {path_manager.current_index}")
        print(f"Current chunk: {chunk}")
        print(type(chunk))
        print(chunk.shape)
        print("Current chunk:")
            
        if path_manager.current_index + path_manager.chunk_size >= len(path_manager.current_path):
            print("\nReached end of path!")
            break
            
        next_point = [current_pos[0] + np.random.uniform(low=0.003, high=0.05, size=1)[0], current_pos[1] + np.random.uniform(low=-0.05, high=0.05, size=1)[0]]
        current_pos = next_point