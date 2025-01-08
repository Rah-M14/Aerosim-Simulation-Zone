import numpy as np
import math
import random


class MovementGenerator:
    """
    Class to generate movement scripts for characters in an episode,
    while avoiding specific rectangular areas.

    Attributes:
        max_steps (int): Maximum number of movement steps for each character.
        group_deviation (float): Maximum deviation for group members from the group center.
        individual_step_size (float): Step size for individual movements.
        min_distance (float): Minimum distance between characters to avoid collisions.
        action_probabilities (dict): Probabilities for different actions.
        obstacles (list): List of rectangular regions to avoid.
    """

    def __init__(
        self,
        # world_graph,
        max_steps: int = 50,
        group_deviation: float = 1,
        individual_step_size: float = 3.0,
        min_distance: float = 0.5,
        action_probabilities=None,
        character_positions=None,
        obstacles=None,
        hotspots=None,
        prims=None,
    ):
        # self.world_graph = world_graph
        self.max_steps = max_steps
        self.group_deviation = group_deviation
        self.individual_step_size = individual_step_size
        self.min_distance = min_distance
        self.obstacles = obstacles if obstacles is not None else []
        self.hotspots = hotspots if hotspots is not None else []
        self.prims = prims if prims is not None else []
        self.character_positions = character_positions
        self.action_probabilities = action_probabilities or {
            "GoTo": 0.7,
            "LookAround": 0.15,
            "Idle": 0.15,
            "Sit": 0.0,
        }
        self.grouped_dict = {}
        self._create_group_dict(character_positions)

    def _centroid_2d(self, points):
        x_coords, y_coords = zip(*points)
        centroid_x = sum(x_coords) / len(points)
        centroid_y = sum(y_coords) / len(points)
        return (centroid_x, centroid_y)

    def _euclidean_hotspot(self, target, p):
        hotspot = (p[1], p[2])
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(target, hotspot)))

    def _closest_hotspot(self, target_point):
        return min(
            self.hotspots, key=lambda p: self._euclidean_hotspot(target_point, p)
        )

    def _choose_action(self):
        actions, probabilities = zip(*self.action_probabilities.items())
        return np.random.choice(actions, p=probabilities)

    def _get_character_position(self, character_name):
        return self.character_positions[character_name]

    def _generate_group_trajectory(self, group_size, members):
        if len(members) > 1:
            action = self._choose_action()
            while action == "Sit":
                action = self._choose_action()
        else:
            action = self._choose_action()
        if action == "GoTo":
            return self._generate_go_to_trajectory(members)
        elif action == "LookAround":
            duration = np.random.randint(4, 8)
            return [f"LookAround {duration}"] * group_size
        elif action == "Idle":
            duration = np.random.randint(4, 8)
            return [f"Idle {duration}"] * group_size
        elif action == "Sit":
            duration = np.random.randint(10, 20)
            random_prim = self.prims[np.random.randint(len(self.prims))]
            return [f"Sit {random_prim[1]} {duration}"]

    def _generate_go_to_trajectory(self, members):
        group_positions = [self._get_character_position(member) for member in members]
        group_centroid = self._centroid_2d(group_positions)
        start_hotspot = self._closest_hotspot(group_centroid)
        next_hotspot = self.hotspots[random.choice(start_hotspot[3]) - 1]
        x_start, y_start, x_next, y_next = (
            start_hotspot[1],
            start_hotspot[2],
            next_hotspot[1],
            next_hotspot[2],
        )
        transformation = np.array((x_next - x_start, y_next - y_start))
        goto_commands = []
        for member in members:
            start_pos = np.array(self._get_character_position(member))
            noise = np.random.normal(0, 0.01, start_pos.shape)
            next_pos = transformation + start_pos + noise
            goto_commands.append(f"GoTo {next_pos[0]:.4f} {next_pos[1]:.4f} 0 _")
            self.character_positions[member] = (next_pos[0], next_pos[1])
        return goto_commands

    def _create_group_dict(self, input_dict):
        for key in input_dict:
            group_number = int(key.split("_G_")[1])
            if group_number not in self.grouped_dict:
                self.grouped_dict[group_number] = []
            self.grouped_dict[group_number].append(key)

    def generate_step(self):
        for group_id, members in self.grouped_dict.items():
            group_size = len(members)
            group_trajectories = self._generate_group_trajectory(
                group_size, members
            )

    def generate_movement_script(self, spawn_commands):
        lines = spawn_commands.splitlines()
        grouped_dict = {}
        for line in lines:
            parts = line.split()
            if parts[0] == "Spawn":
                character_name = parts[1]
            else:
                character_name = parts[0]
            if "_G_" in character_name:
                group_id = character_name.split("_G_")[1]
                grouped_dict.setdefault(group_id, []).append(character_name)

        all_actions = []
        for _ in range(self.max_steps):
            for group_id, members in grouped_dict.items():
                group_size = len(members)
                group_trajectories = self._generate_group_trajectory(
                    group_size, members
                )
                for member, trajectory in zip(members, group_trajectories):
                    all_actions.append(f"{member} {trajectory}")
            all_actions.append("\n")

        return "\n".join(all_actions)
