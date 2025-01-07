import numpy as np
import random
import copy

class CreatePeople:
    """
    Class to generate spawn points for people with random locations, rotations, and groupings,
    while avoiding specific rectangular areas and generating points around distinct hotspots for each group.
    """

    def __init__(
        self,
        min_people: int,
        max_people: int,
        hotspots: list,  # Hotspots for generating points around
        min_delta: float,
        max_delta: float,
        max_group_share: float,
        max_group_size: int,
        obstacles: list,
        sigma: float,  # Standard deviation for Gaussian distribution
    ) -> None:
        self.min_people = min_people
        self.max_people = max_people
        self.hotspots = copy.deepcopy(hotspots)
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.max_group_share = max_group_share
        self.max_group_size = max_group_size
        self.obstacles = obstacles
        self.sigma = sigma
        self.spawn_commands = []
        self.spawn_positions = {}

    def is_inside_obstacles(self, x: float, y: float) -> bool:
        """
        Check if a point (x, y) is inside any of the obstacles.

        Args:
            x (float): x-coordinate of the point.
            y (float): y-coordinate of the point.

        Returns:
            bool: True if the point is inside any rectangle, False otherwise.
        """
        for x1, y1, x2, y2 in self.obstacles:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False
    
    def _random_hotspot(self):
        weights = [item[4] for item in self.hotspots]
        chosen_hotspot = random.choices(self.hotspots, weights=weights, k=1)[0]
        return chosen_hotspot

    def generate_random_point(self, hotspot) -> tuple:
        """
        Generate a random point around a given hotspot using a Gaussian distribution,
        avoiding obstacles.

        Args:
            hotspot (tuple): The (x, y) coordinates of the hotspot.

        Returns:
            tuple: (x, y) coordinates of the generated point.
        """
        while True:
            x_hotspot, y_hotspot = hotspot[1], hotspot[2]
            x = np.random.normal(x_hotspot, self.sigma)
            y = np.random.normal(y_hotspot, self.sigma)

            if not self.is_inside_obstacles(x, y):
                return (x, y)
            
    def _create_charcaters(self, num_persons, num_grouped_people, ungrouped_people):
        group_dict = {}

        group_number = 1
        person_number = 1
        grouped_people_count = 0

        # Grouped people
        while grouped_people_count < num_grouped_people:
            current_group_size = min(self.max_group_size, num_grouped_people - grouped_people_count)
            
            # Initialize list for current group
            group_dict[group_number] = []
            
            for i in range(current_group_size):
                character_name = f"P_{person_number}_G_{group_number}"
                group_dict[group_number].append(character_name)
                person_number += 1
            
            grouped_people_count += current_group_size
            group_number += 1

        # Ungrouped people (if any)
        if ungrouped_people > 0:
            for i in range(ungrouped_people):
                group_dict[group_number] = []  # start a new group for ungrouped people
                character_name = f"P_{person_number}_G_{group_number}"
                group_dict[group_number].append(character_name)
                person_number += 1
                group_number += 1
        return group_dict

            
    def _create_locations(self, group):
        num_points = len(group)
        choice = self._random_hotspot()
        for point in range(num_points):
            r = np.random.uniform(0, 180)
            x, y = self.generate_random_point(choice)
            self.spawn_commands.append(f"Spawn {group[point]} {x:.4f} {y:.4f} 0 {r:.4f}")
            self.spawn_positions[f"{group[point]}"] = (x, y)
        self.hotspots.remove(choice)

    def generate_spawns(self, return_commands=True) -> str:
        """
        Generate random spawn commands for a number of people between min_people and max_people,
        with grouping constraints, while assigning different groups to different hotspots.

        Returns:
            str: Spawn commands formatted as "Spawn P_i_G_j x y z r" with newline separation.
        """
        num_persons = np.random.randint(self.min_people, self.max_people)
        num_grouped_people = int(num_persons * self.max_group_share)
        ungrouped_people = num_persons - num_grouped_people
        
        charcaters_dict = self._create_charcaters(num_persons, num_grouped_people, ungrouped_people)
        
        for group in charcaters_dict:
            self._create_locations(charcaters_dict[group])

        # spawn_commands = []
        # group_id = 1
        # person_id = 1
        # hotspot_indices = np.random.permutation(len(self.hotspots))
        # spawn_positions = {}

        # while num_grouped_people > 0:
        #     hotspot = self._random_hotspot()
        #     self.hotspots.remove(hotspot)
        #     # hotspot = self.hotspots[hotspot_indices[group_id - 1]]
        #     group_size = min(
        #         np.random.randint(2, self.max_group_size + 1), num_grouped_people
        #     )
        #     x_base, y_base = self.generate_random_point(hotspot)

        #     for _ in range(group_size):
        #         x = x_base + np.random.uniform(self.min_delta, self.max_delta) * (
        #             -1 if np.random.random() < 0.5 else 1
        #         )
        #         y = y_base + np.random.uniform(self.min_delta, self.max_delta) * (
        #             -1 if np.random.random() < 0.5 else 1
        #         )
        #         r = np.random.uniform(0, 180)

        #         if not self.is_inside_obstacles(x, y):
        #             spawn_command = (
        #                 f"Spawn P_{person_id}_G_{group_id} {x:.4f} {y:.4f} 0 {r:.4f}"
        #             )
        #             spawn_positions[f"P_{person_id}_G_{group_id}"] = (x, y)
        #             spawn_commands.append(spawn_command)

        #             person_id += 1
        #             num_grouped_people -= 1

        #     group_id += 1

        # for _ in range(ungrouped_people):
        #     hotspot = self._random_hotspot()
        #     self.hotspots.remove(hotspot)
        #     x, y = self.generate_random_point(hotspot)
        #     r = np.random.uniform(0, 180)

        #     spawn_command = (
        #         f"Spawn P_{person_id}_G_{group_id} {x:.4f} {y:.4f} 0 {r:.4f}"
        #     )
        #     spawn_positions[f"P_{person_id}_G_{group_id}"] = (x, y)
        #     spawn_commands.append(spawn_command)

        #     person_id += 1
        #     group_id += 1

        if return_commands == True:
            return "\n".join(self.spawn_commands), self.spawn_positions
        else:
            return self.spawn_positions
