import carb
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

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

from Traj_Gen.create_people import CreatePeople
from Traj_Gen.move_generator import MovementGenerator

import json

class CustomArguments:
    def __init__(
        self,
        world_number=1,
        min_people=15,
        max_people=20,
        num_episodes=1,
        max_group_share=0.5,
        max_group_size=3,
        min_delta=0.35,
        max_delta=0.8,
        offset=0.2,
        output_dir="outputs",
        max_steps=500,
        group_deviation=0.5,
        individual_step_size=2.0,
        interpolation=100,
    ):
        self.worlds_file = "./standalone_examples/api/omni.isaac.kit/MAIN_FILES/Traj_Gen/worlds.json"
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

class GoTo_Controller(PersonController):

    def __init__(self, movement_generator):
        super().__init__()

        # self.person_pos = self._person._state.position
        self.next_position = None
        self.speed = 0.3

        self.movement_generator = movement_generator

    def update(self, dt=1/2):
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
        
        # if character_name == "P_1_G_1":
        #     print("inside update")
        #     print(f"person 1 g1 position : {self._person._state.position}")

        if distance_to_target_position > 0.2:
            pass
        elif distance_to_target_position < 0.2:
            if character_name == "P_1_G_1":
                self.movement_generator.generate_step()
            self.next_position = self.movement_generator.character_positions[character_name] + (0,)
            self._person.update_target_position(self.next_position, self.speed + np.random.uniform(low=-0.1, high=0.1, size=1)[0])

class PegasusApp:
    def __init__(self, world, stage, simulation_app, timeline):

        args = CustomArguments()
        self.args = args

        self.simulation_app = simulation_app
        self.person_list = []

        self.stage = stage
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

        # root_path = "/World/Characters"
        # character_prim_paths = []
        # for char in spawn_positions.items():
        #     character_prim_paths.append(root_path + "/" + char[0])

        # print("Character Prim Paths: ", character_prim_paths)

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

        # for char in character_prim_paths:
        #     rigid_prim = self.stage.GetPrimAtPath(char)
        #     rigid_API = UsdPhysics.RigidBodyAPI.Apply(rigid_prim)
        #     rigid_API.CreateKinematicEnabledAttr(True)
            
        #     coll_obj = char + "/Collider_Cylinder"
        #     coll_geom = UsdGeom.Cylinder.Define(self.stage, coll_obj)
        #     coll_prim = self.stage.GetPrimAtPath(coll_obj)
        #     coll_geom.CreateHeightAttr(3.0)
        #     coll_geom.CreateRadiusAttr(0.25)
        #     coll_API = UsdPhysics.CollisionAPI.Apply(coll_prim)
        #     coll_geom.CreatePurposeAttr(UsdGeom.Tokens.guide)
        #     coll_API.CreateCollisionEnabledAttr(True)
        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def _read_json(self, args):
        # Unpacking arguments
        world_number = args.world_number

        with open(args.worlds_file) as f:
            data = json.load(f)

        world = data["worlds"][world_number]

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

    # def run(self):
    #     """
    #     Method that implements the application main loop, where the physics steps are executed.
    #     """
    #     self.timeline.play()

    #     i = 0
    #     while self.simulation_app.is_running() and not self.stop_sim:
    #         self.world.step(render=True)
    #         print("in pegasus run")
    #         i+=1

    #     # Cleanup and stop
    #     carb.log_warn("PegasusApp Simulation App is closing.")
    #     self.timeline.stop()
    #     self.simulation_app.close()
