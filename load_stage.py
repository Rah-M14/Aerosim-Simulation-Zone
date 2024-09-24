# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import argparse
import sys

from isaacsim import SimulationApp

# This sample loads a usd stage and starts simulation
CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}

# Set up command line arguments
parser = argparse.ArgumentParser("Usd Load sample")
parser.add_argument(
    "--usd_path", type=str, help="Path to usd file, should be relative to your default assets folder", required=True
)
parser.add_argument("--headless", default=False, action="store_true", help="Run stage headless")
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")

args, unknown = parser.parse_known_args()
# Start the omniverse application
CONFIG["headless"] = args.headless
kit = SimulationApp(launch_config=CONFIG)

import carb
import omni

# Locate Isaac Sim assets folder to load sample
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers import DifferentialController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.controllers import BaseController
from omni.isaac.nucleus import get_assets_root_path, is_file
from pxr import Usd, Vt

import numpy as np

class OpenController(BaseController):
    def __init__(self):
        super().__init__(name="open_controller")
        # An open loop controller that uses a unicycle model
        self._wheel_radius = 0.03
        self._wheel_base = 0.1125
        return

    def forward(self, command):
        # command will have two elements, first element is the forward velocity
        # second element is the angular velocity (yaw only).
        joint_velocities = [0.0, 0.0]
        joint_velocities[0] = ((2 * command[0]) - (command[1] * self._wheel_base)) / (2 * self._wheel_radius)
        joint_velocities[1] = ((2 * command[0]) + (command[1] * self._wheel_base)) / (2 * self._wheel_radius)
        # A controller has to return an ArticulationAction
        return ArticulationAction(joint_velocities=np.array(joint_velocities))

def send_robot_actions(bot, controller, v, w):
    position, orientation = bot.get_world_pose()
    print(f"Bot's Position : {position}, Bot's Orientation : {orientation} ")
    #apply the actions calculated by the controller
    bot.apply_action(controller.forward(command=[v, w]))
    return

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    kit.close()
    sys.exit()
usd_path = assets_root_path + args.usd_path

# make sure the file exists before we try to open it
try:
    result = is_file(usd_path)
except:
    result = False

if result:
    # omni.usd.get_context().open_stage(usd_path)
    stage = Usd.Stage.Open('omniverse://localhost/Projects/SIMS/PEOPLE_SIMS/New_Core.usd')
else:
    carb.log_error(
        f"the usd path {usd_path} could not be opened, please make sure that {args.usd_path} is a valid usd file in {assets_root_path}"
    )
    kit.close()
    sys.exit()
# Wait two frames so that stage starts loading
kit.update()
kit.update()

print("Loading stage...")
from omni.isaac.core.utils.stage import is_stage_loading

while is_stage_loading():
    kit.update()
print("Loading Complete")
world = World()
print("got world")

jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
world.scene.add(
    WheeledRobot(
        prim_path="/World/Jetbot",
        name="Jetbot",
        wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        create_robot=True,
        usd_path=jetbot_asset_path,
        position=np.array([0.4, -0.4,0])
    )
)
jetbot = ArticulationView(prim_paths_expr="/World/Jetbot" , name="Jetbot")
# jetbot.initialize()

world.initialize_physics()
jetbot = world.scene.get_object("Jetbot")

print('Bot Created!')
# world.add_physics_callback("sending_actions", callback_fn=send_robot_actions)
        # Initialize our controller after load and the first reset
# my_controller = OpenController()
controller=DifferentialController(name="simple_control", wheel_radius=0.035, wheel_base=0.1125)

throttle = 1.0
steering = 0.5
# jetbot.apply_wheel_actions(my_controller.forward(throttle, steering))
print("Controller Created!")

for i in range(5):
    position, orientation = jetbot.get_world_pose()
    linear_velocity = jetbot.get_linear_velocity()
    # will be shown on terminal
    print("Bot's position is : " + str(position))
    print("Bot's orientation is : " + str(orientation))
    print("Bot's linear velocity is : " + str(linear_velocity))
    # rnd = np.random.randint(1,5)
    if i == 0:
        jetbot.apply_wheel_actions(controller.forward(np.array([0,0])))
        print("Steady!")
    elif i == 1: # forward
        jetbot.apply_wheel_actions(controller.forward(np.array([throttle, 0])))
        print(f"Angular Vel : {jetbot.get_angular_velocity()}")
        print(f"applied action : {jetbot.get_applied_action()}")
        # jetbot.apply_wheel_actions(my_controller.forward(throttle, steering))
        # jetbot.apply_wheel_actions(my_controller.forward(np.array(throttle, steering)))
        # send_robot_actions(jetbot, my_controller, throttle, 0)
        print("Action Forward")
    elif i == 2: # backward
        jetbot.apply_wheel_actions(controller.forward(np.array([-throttle, 0])))
        # send_robot_actions(jetbot, my_controller, i, np.pi)
        print("Action Backward!")
    elif i == 3: # left
        jetbot.apply_wheel_actions(controller.forward(np.array([0, steering])))
        # send_robot_actions(jetbot, my_controller, i, np.pi/2)
        print("Action Left!")
    elif i == 4: # right
        jetbot.apply_wheel_actions(controller.forward(np.array([0, -steering])))
        # send_robot_actions(jetbot, my_controller, i, -np.pi/2)
        print("Action Right!")
    # we have control over stepping physics and rendering in this workflow
    # things run in sync
    world.step(render=True) # execute one physics step and one rendering step

print('loop is done!')

omni.timeline.get_timeline_interface().play()
# Run in test mode, exit after a fixed number of steps
if args.test is True:
    for i in range(10):
        # Run in realtime mode, we don't specify the step size
        kit.update()
else:
    while kit.is_running():
        # Run in realtime mode, we don't specify the step size
        kit.update()

# omni.timeline.get_timeline_interface().stop()


