import numpy as np
from RRTStar import RRTStarPlanner, get_final_path

class PathManager:
    def __init__(self, image_path, chunk_size=10, overlap_ratio=0.5):
        self.planner = RRTStarPlanner(image_path)
        self.chunk_size = chunk_size
        self.overlap_size = int(chunk_size * overlap_ratio)
        self.current_path = None
        self.current_index = 0
        self.new_chunk = None
        self.start = None
        self.goal = None
        
    def plan_new_path(self, start_pos, goal_pos):
        self.start = start_pos
        self.goal = goal_pos
        result = self.planner.plan(start_pos, goal_pos)
        self.current_path = result['path']
        self.current_index = 0
        self.new_chunk = np.array(self.current_path[self.current_index:self.current_index + self.chunk_size])
        return result
    
    def get_next_chunk(self, current_pos):
        if not self.needs_new_chunk(current_pos):
            return np.array(self.new_chunk)
        
        self.current_index += self.get_chunk_progress(current_pos)
        if self.current_index + self.chunk_size < len(self.current_path):
            self.new_chunk = self.current_path[self.current_index:self.current_index + self.chunk_size]
        else:
            rem_path = self.current_path[self.current_index:]
            final_element = self.current_path[-1]
            padding = [final_element] * (self.chunk_size - len(rem_path))
            self.new_chunk = np.vstack((rem_path, padding))
        return np.array(self.new_chunk)
    
    def get_chunk_progress(self, current_pos):
        if (current_pos==self.current_path[0]).all():
            return 0
        
        min_dist = float('inf')
        closest_idx = 0
        for i, coords in enumerate(self.new_chunk):
            dist = np.linalg.norm(current_pos - coords)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        return closest_idx

    def needs_new_chunk(self, current_pos):
        if self.new_chunk is None:
            return True
        
        progress_idx = self.get_chunk_progress(current_pos)
        return progress_idx >= self.overlap_size and self.current_index < len(self.current_path)
    
    def reset(self):
        self.current_index = 0
        self.new_chunk = None

    def get_full_path(self):
        return self.current_path if self.current_path else []

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