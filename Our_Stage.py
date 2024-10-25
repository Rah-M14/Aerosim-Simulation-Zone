#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) # we can also run as headless.

import omni

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController

from New_RL_Bot import RLBot
from New_RL_Bot_Control import RLBotController, RLBotAct
import numpy as np

class TestBot():
    def __init__(self):
        self.bot = None
        return

    def setup_scene(self):
        self.world = World()
        self.world.scene.add_default_ground_plane()
        assets_root_path = get_assets_root_path()
        timeline = omni.timeline.get_timeline_interface() 

        self.bot = RLBot(simulation_app, self.world, timeline, assets_root_path, "carter")
        # self.core_controller = RLBotController("jackal")
        self.controller = WheelBasePoseController(name='Path_Bot_Controller',
                                                    open_loop_wheel_controller=DifferentialController(name="simple_control",
                                                                                    wheel_radius=0.24, wheel_base=0.54),
                                                    is_holonomic=False)
        # self.bot_act = RLBotAct(self.bot.rl_bot, self.bot_controller, n_steps=1)
        self.world.reset()
        return
    
    def run(self):
        position = np.array([0.0, 0.0, 0.0])
        orientation = np.array([1, 0.0, 0.0, 0.0])
        self.bot.rl_bot.apply_wheel_actions(self.controller.forward(start_position=position,
                                                            start_orientation=orientation,
                                                            goal_position=np.array([5.0, 0])))
        # command = np.array([1.0, 0.0])
        # self.bot_act.move_bot(command)
        self.world.step(render=True)
        return

    def close(self):
        simulation_app.close()
        return

if "__main__" == __name__:
    app = TestBot()
    # app.init()
    print("TestBot initialized")
    app.setup_scene()
    app.run()
    while True:
        print(f"{app.bot.rl_bot.get_world_pose()[0]}")
        simulation_app.update()
    # app.close()

# world = World()
# world.scene.add_default_ground_plane()
# fancy_cube =  world.scene.add(
#     DynamicCuboid(
#         prim_path="/World/random_cube",
#         name="fancy_cube",
#         position=np.array([0, 0, 1.0]),
#         scale=np.array([0.5015, 0.5015, 0.5015]),
#         color=np.array([0, 0, 1.0]),
#     ))
# # Resetting the world needs to be called before querying anything related to an articulation specifically.
# # Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
# world.reset()
# for i in range(500):
#     position, orientation = fancy_cube.get_world_pose()
#     linear_velocity = fancy_cube.get_linear_velocity()
#     # will be shown on terminal
#     print("Cube position is : " + str(position))
#     print("Cube's orientation is : " + str(orientation))
#     print("Cube's linear velocity is : " + str(linear_velocity))
#     # we have control over stepping physics and rendering in this workflow
#     # things run in sync
#     world.step(render=True) # execute one physics step and one rendering step

# simulation_app.close() # close Isaac Sim