
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
import matplotlib.pyplot as plt
import numpy as np
from pxr import Usd, Vt, Gf

import omni.kit.viewport.utility
from omni.isaac.core import World, SimulationContext
from omni.isaac.core.utils import stage
from omni.isaac.core.scenes import Scene
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers import DifferentialController, WheelBasePoseController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.controllers import BaseController
from omni.isaac.sensor import Camera
from omni.isaac.sensor import LidarRtx
# from omni.isaac.range_sensor import _range_sensor 
from omni.isaac.range_sensor._range_sensor import acquire_lidar_sensor_interface
from omni.isaac.nucleus import get_assets_root_path, is_file
from omni.isaac.core.utils.extensions import enable_extension

import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.replicator.core as rep
import omni.timeline
from omni.isaac.core.world import World
from omni.isaac.core.utils.extensions import disable_extension, enable_extension
import time

EXTENSIONS_PEOPLE = [
    'omni.anim.people', 
    'omni.anim.navigation.bundle', 
    'omni.anim.timeline',
    'omni.anim.graph.bundle', 
    'omni.anim.graph.core', 
    'omni.anim.graph.ui',
    'omni.anim.retarget.bundle', 
    'omni.anim.retarget.core',
    'omni.anim.retarget.ui', 
    'omni.kit.scripting',
    'omni.graph.io',
    'omni.anim.curve.core',
]

for ext_people in EXTENSIONS_PEOPLE:
    enable_extension(ext_people)

# Update the simulation app with the new extensions
kit.update()

from Pegasus_App import PegasusApp

# CUSTOM_CONTROLLER CONTROL
class CarterController(BaseController):
    def __init__(self):
        super().__init__(name="carter_controller")
        self._wheel_radius = 0.41
        self._wheel_base = 0.413
        return

    def forward(self, command):
        # command will have two elements, first element is the forward velocity & second element is the angular velocity (yaw only).
        joint_velocities = [0.0, 0.0, 0.0, 0.0]
        joint_velocities[0] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        joint_velocities[1] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        joint_velocities[2] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        joint_velocities[3] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        return ArticulationAction(joint_velocities=joint_velocities)

# CONTROLLER CONTROL
def go_forward(bot, controller, n_steps=5):
    for _ in range(n_steps):
        bot.apply_wheel_actions(controller.forward(np.array([1, 0])))
    return None

def go_backward(bot, controller, n_steps=5):
    for _ in range(n_steps):
        bot.apply_wheel_actions(controller.forward(np.array([-1, 0])))
    return None

def turn_right(bot, controller, n_steps=20):
    for _ in range(n_steps):
        bot.apply_wheel_actions(controller.forward(np.array([0, -np.pi/2])))
    return None

def turn_left(bot, controller, n_steps=20):
    for _ in range(n_steps):
        bot.apply_wheel_actions(controller.forward(np.array([0, np.pi/2])))
    return None

def stay(bot, controller, n_steps=5):
    for _ in range(n_steps):
        bot.apply_wheel_actions(controller.forward(np.array([0, 0])))
    return None

def bot_act(bot, controller, action, n_steps=5):
    if action == 0:
        stay(bot, controller, n_steps)
    elif action == 1:
        go_forward(bot, controller, n_steps)
    elif action == 2:
        go_backward(bot, controller, n_steps)
    elif action == 3:
        turn_right(bot, controller, n_steps)
    elif action == 4:
        turn_left(bot, controller, n_steps)
    return None

act_dict = {0 : 'stay', 1 : 'forward', 2 : 'backward', 3 : 'right', 4 : 'left'}

# THE WORLD SECTION STARTS HERE

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
    omni.usd.get_context().open_stage(usd_path)
    wr_stage = omni.usd.get_context().get_stage()

    # stage = Usd.Stage.Open('omniverse://localhost/Projects/SIMS/PEOPLE_SIMS/New_Core.usd')
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

wr_world = World()
print("Waiting Room in the World!")

enable_extension("omni.isaac.debug_draw")
lidar_config = 'RPLIDAR_S2E'
wr_timeline = omni.timeline.get_timeline_interface() 
wr_world.initialize_physics()
print("Physics Initialized!")

# CARTER
carter_asset_path = assets_root_path + "/Isaac/Robots/Carter/nova_carter_sensors.usd"
carter_bot = WheeledRobot(
        prim_path="/World/Nova_Carter",
        name="Carter",
        wheel_dof_names=["joint_caster_left", "joint_caster_right", "joint_wheel_left", "joint_wheel_right"],
        wheel_dof_indices=[3, 4, 5, 6],
        create_robot=True,
        usd_path=carter_asset_path,
        position=np.array([0.4, -0.4,0]),
    )
wr_world.scene.add(carter_bot) 
# carter_bot.enable_gravity()
# carter_bot.set_default_state(position=np.array([-4, -4, 0]))
# carter_bot.set_enabled_self_collisions(True)

carter = ArticulationView(prim_paths_expr="/World/Nova_Carter", name="Carter")
carter = wr_world.scene.get_object("Carter")
print(f"Carter : {carter}")
print("Carter in the World")

# CARTER CAMERA
carter_camera = Camera(prim_path="/World/Nova_Carter/chassis_link/front_owl/camera", 
                    name='Carter_Camera',
                    frequency=30,
                    resolution=(512,512))
carter_camera.initialize()
kit.update()
carter_camera.initialize()
print(f"Carter Camera : {carter_camera}")
wr_world.initialize_physics()

# CARTER LiDAR
carter_lidar = wr_world.scene.add(
    LidarRtx(prim_path="/World/Nova_Carter/chassis_link/front_RPLidar/RPLIDAR_S2E", 
             name="Carter_Lidar"))
print(f"Carter LiDAR : {carter_lidar}")
print("performing reset")
wr_world.reset()
print("reset done")
carter_lidar.add_range_data_to_frame()
carter_lidar.add_point_cloud_data_to_frame()
carter_lidar.enable_visualization()

# CARTER CONTROLLER
carter_controller = CarterController()
print("Controller Created!")

for i in range(2000):
    kit.update()
print("launching pegasus")

pg_app = PegasusApp(wr_world, wr_timeline, kit)
pg_app.run()


i = 0
reset_needed = False
while kit.is_running():
    wr_world.step(render=True)
    if wr_world.is_stopped() and not reset_needed:
        reset_needed = True
    if wr_world.is_playing():
        if reset_needed:
            wr_world.reset()
            # bot_controller.reset()
            reset_needed = False
        if i >= 0:
            with open('/home/rah_m/Isaac_World_Files/lidar_data.txt', 'a+') as f:
                f.write(str(carter_lidar.get_current_frame()))
                f.write('\n')
            if i % 200 == 0:
                act_val = np.random.randint(0,5,1)
                # act_val = np.array([1])
                print(f"Action Value : {act_val} - {act_dict[act_val[0]]}")
                bot_act(carter, carter_controller, act_val)
                with open('/home/rah_m/Isaac_World_Files/camera_data.txt', 'a+') as f:
                    f.write(str(carter_camera.get_current_frame()))
                    f.write('\n')
                # print(carter_camera.get_current_frame())    
                imgplot = plt.imshow(carter_camera.get_rgba()[:, :, :3])
                plt.show()
                # print(carter_camera.get_current_frame()["motion_vectors"])
            # print(carter_lidar.get_current_frame())
            # forward
            # carter.apply_wheel_actions(bot_controller.forward(command=[0.0, np.pi/2]))
        else:
            break
        # elif i >= 1000 and i < 1265:
        #     # rotate
        #     carter.apply_wheel_actions(bot_controller.forward(command=[0.0, np.pi / 12]))
        # elif i >= 1265 and i < 2000:
        #     # forward
        #     carter.apply_wheel_actions(bot_controller.forward(command=[0.05, 0]))
        # elif i == 2000:
        #     i = 0
        i += 1
wr_world.stop()
print('loop is done!')

# omni.timeline.get_timeline_interface().play()
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