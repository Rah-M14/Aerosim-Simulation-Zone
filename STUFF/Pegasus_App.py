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

# import omni.anim.graph.core as ag

class CirclePersonController(PersonController):

    def __init__(self):
        super().__init__()

        self._radius = 5.0
        self.gamma = 0.0
        self.gamma_dot = 0.3
        
    def update(self, dt: float):

        # Update the reference position for the person to track
        self.gamma += self.gamma_dot * dt
        
        # Set the target position for the person to track
        self._person.update_target_position([self._radius * np.cos(self.gamma), self._radius * np.sin(self.gamma), 0.0])

class GoTo_Controller(PersonController):

    def __init__(self):
        super().__init__()

        self.next_position = None
        self.speed = 1.0
        
    def update(self, dt: float):

        # print(f"Character accessed : {self._person._character_name}")
        # print(f"Character_Stage_Name accessed : {self._person._stage_prefix}")
        # It is "/World/Characters/{THE_NAME}"

        # self.current_position = self._person._state.position
        # print(f" Current Position : {self._person._state.position}")

        if self.next_position is None:
            distance_to_target_position = 0.0
        else:
            distance_to_target_position = np.linalg.norm(self.next_position - self._person._state.position)
            # print(f"Current_position in D : ", self._person._state.position)
            # print(f"Next_position in D : ", self.next_position)
            # print(f"Distance to target position : {distance_to_target_position}")

        if distance_to_target_position > 0.2:
            # print("Walking")
            # print(f"Next position : {self.next_position}")
            self._person.update_target_position(self.next_position, self.speed)
        elif distance_to_target_position < 0.2:
            # self._person._state.position = self.next_position
            self.random_pos = np.random.uniform(-7, 7, 3)
            self.speed = np.random.choice([0.5, 5])
            self.random_pos[-1] = 0.0
            self.next_position = self.random_pos
            # print(f"Next position : {self.next_position}")
            self._person.update_target_position(self.next_position, self.speed)
            # print("Updating to Target Position")

# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation

class PegasusApp:
    def __init__(self, world, timeline, simulation_app):
        self.simulation_app = simulation_app
        self.person_list = []

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = timeline

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics,
        # spawning asset primitives, etc.
        self.pg._world = world
        self.world = self.pg.world

        # Check the available assets for people
        people_assets_list = Person.get_character_asset_list()
        asset_list = ["female_adult_police_01_new",
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
                    "original_male_adult_police_04"]
        
        for asset in people_assets_list:
            asset_list.append(asset)

        # PeopleManager.get_people_manager().rebuild_nav_mesh(radius=14, auto_rebake_on_changes=True, 
        #                             auto_rebake_delay_seconds=4, exclude_rigid_bodies=False, 
        #                             view_nav_mesh=True, dynamic_avoidance_enabled=True, 
        #                             navmesh_enabled=True)
        
        # Create the controller to make on person walk around in circles
        # person_controller = CirclePersonController()
        # p1 = Person("person1", "original_male_adult_construction_05", init_pos=[3.0, 0.0, 0.0], init_yaw=1.0, controller=person_controller)
        
        for i in range(10):
            p = Person(f"person{i}", asset_list[i], init_pos=[3.0, 0.0, 0.0], controller=GoTo_Controller())
            self.person_list.append(p)

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def people_goto(self):
        for i,p in enumerate(self.person_list):
            p.update_target_position([5.0+i, 4.0-i, 0.0], 1.0)
    
    def people_goto_2(self):
        for i,p in enumerate(self.person_list):
            p.update_target_position([5.0-i, 4.0+i, 0.0], 1.0)
    
    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """
        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        i = 0
        while self.simulation_app.is_running() and not self.stop_sim:
            # Update the UI of the app and perform the physics step
            # for p in self.person_list:
            #     p.update_target_position([5.0, 5.0, 0.0], 1.0)
            self.world.step(render=True)
            # if i == 0:
            #     self.people_goto()
            #     self.world.step(render=True)
            # elif i == 15:
            #     self.people_goto_2()
            #     self.world.step(render=True)
            # i+=1
                
        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        self.simulation_app.close()
