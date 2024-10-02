import carb
import numpy as np

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.people.person import Person
from pegasus.simulator.logic.people_manager import PeopleManager
from pegasus.simulator.logic.people.person_controller import PersonController

# from pegasus.simulator.logic.backends.mavlink_backend import MavlinkBackend, MavlinkBackendConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from scipy.spatial.transform import Rotation

import omni.anim.graph.core as ag

from Trajectory_Gen_2.create_people import CreatePeople
from Trajectory_Gen_2.move_generator import MovementGenerator

import json

class CustomArguments:
    def __init__(
        self,
        world_number=1,
        min_people=10,
        max_people=20,
        num_episodes=1,
        max_group_share=0.5,
        max_group_size=4,
        min_delta=0.35,
        max_delta=0.8,
        offset=0.2,
        output_dir="outputs",
        max_steps=500,
        group_deviation=0.5,
        individual_step_size=2.0,
        interpolation=100,
    ):
        self.worlds_file = "./standalone_examples/api/omni.isaac.kit/Trajectory_Gen_2/worlds.json"
        self.world_number = world_number
        self.min_people = min_people
        self.max_people = max_people
        self.num_episodes = num_episodes
        self.max_group_share = max_group_share
        self.max_group_size = max_group_size
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.offset = offset
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.group_deviation = group_deviation
        self.individual_step_size = individual_step_size
        self.interpolation = interpolation


# class CirclePersonController(PersonController):

#     def __init__(self):
#         super().__init__()

#         self._radius = 5.0
#         self.gamma = 0.0
#         self.gamma_dot = 0.3

#     def update(self, dt: float):

#         # Update the reference position for the person to track
#         self.gamma += self.gamma_dot * dt

#         # Set the target position for the person to track
#         self._person.update_target_position(
#             [self._radius * np.cos(self.gamma), self._radius * np.sin(self.gamma), 0.0]
#         )


class GoTo_Controller(PersonController):

    def __init__(self, movement_generator):
        super().__init__()

        self.next_position = None
        self.speed = 1.0

        self.movement_generator = movement_generator

    def update(self, dt: float):
        """
        New additions to integrate with previous code.
        """
        character_name = self._person._stage_prefix.split("/")[-1]

        if self.next_position is None:
            distance_to_target_position = 0.0
        else:
            distance_to_target_position = np.linalg.norm(
                self.next_position - self._person._state.position
            )

        if distance_to_target_position > 0.2:
            pass
        elif distance_to_target_position < 0.2:
            if character_name == "P_1_G_1":
                self.movement_generator.generate_step()
            self.next_position = self.movement_generator.character_positions[character_name] + (0,)
            self._person.update_target_position(self.next_position, self.speed)

class PegasusApp:
    def __init__(self, world, timeline, simulation_app):

        args = CustomArguments()
        self.args = args

        self.simulation_app = simulation_app
        self.person_list = []

        self.timeline = timeline

        self.pg = PegasusInterface()

        self.pg._world = world
        self.world = self.pg.world

        people_assets_list = Person.get_character_asset_list()
        asset_list = [
            "female_adult_police_01_new",
            "female_adult_police_02",
            "female_adult_police_03_new",
            "male_adult_construction_01_new",
            "male_adult_construction_03",
            "male_adult_construction_05_new",
            "male_adult_police_04",
            "original_female_adult_business_02",
            "original_female_adult_medical_01",
            "original_female_adult_police_01",
            "original_female_adult_police_02",
            "original_female_adult_police_03",
            "original_male_adult_construction_01",
            "original_male_adult_construction_02",
            "original_male_adult_construction_03",
            "original_male_adult_construction_05",
            "original_male_adult_medical_01",
            "original_male_adult_police_04",
        ]

        for asset in people_assets_list:
            asset_list.append(asset)

        """
            New additions to integrate with previous code.
        """

        self.hotspots_list, self.obstacle_list, self.prims_list = self._read_json(args)

        # Generate the initial spawn commands
        people_creator = CreatePeople(
            min_people=args.min_people,
            max_people=args.max_people,
            hotspots=self.hotspots_list,
            min_delta=args.min_delta,
            max_delta=args.max_delta,
            max_group_share=args.max_group_share,
            max_group_size=args.max_group_size,
            obstacles=self.obstacle_list,
            sigma=0.5,
        )
        spawn_positions = people_creator.generate_spawns(return_commands=False)

        movement_generator = MovementGenerator(
            max_steps=args.max_steps,
            group_deviation=args.group_deviation,
            obstacles=self.obstacle_list,  # Pass obstacles to MovementGenerator
            hotspots=self.hotspots_list,
            prims=self.prims_list,
            character_positions=spawn_positions,
            individual_step_size=args.individual_step_size,
        )

        i = 0
        for person, init_pos in spawn_positions.items():
            p = Person(
                person,
                asset_list[i % len(asset_list)],
                init_pos=init_pos + (0,),
                controller=GoTo_Controller(movement_generator),
            )
            i += 1
            self.person_list.append(p)

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def _read_json(self, args):
        # Unpacking arguments
        world_number = args.world_number

        with open(args.worlds_file) as f:
            data = json.load(f)

        world = data["worlds"][world_number]
        # world_graph = WorldGraph(world) # Not required currently

        obstacles = world["obstacles"]
        obstacle_list = [
            (
                obs["corner1"]["x"],
                obs["corner1"]["y"],
                obs["corner2"]["x"],
                obs["corner2"]["y"],
            )
            for obs in obstacles
        ]

        hotspots = world["hotspots"]
        hotspots_list = [
            (hot["id"], hot["x"], hot["y"], hot["neighbors"], hot["weight"])
            for hot in hotspots
        ]

        prims = world["prims"]
        prims_list = [(prim["id"], prim["prim"]) for prim in prims]

        return hotspots_list, obstacle_list, prims_list

    # def people_goto(self):
    #     for i, p in enumerate(self.person_list):
    #         p.update_target_position([5.0 + i, 4.0 - i, 0.0], 1.0)

    # def people_goto_2(self):
    #     for i, p in enumerate(self.person_list):
    #         p.update_target_position([5.0 - i, 4.0 + i, 0.0], 1.0)

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """
        self.timeline.play()

        i = 0
        while self.simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
            i+=1

        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        self.simulation_app.close()
