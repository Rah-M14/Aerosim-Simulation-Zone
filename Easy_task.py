
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
import stable_baselines3 as sb3

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
from omni.isaac.range_sensor._range_sensor import acquire_lidar_sensor_interface
from omni.isaac.nucleus import get_assets_root_path, is_file
from omni.isaac.core.utils.extensions import enable_extension

import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.replicator.core as rep

print("SB3 imported!")








# CUSTOM DEFINITION OF CLASSES & FUNCTIONS

# class OpenController(BaseController):
#     def __init__(self):
#         super().__init__(name="open_controller")
#         # An open loop controller that uses a unicycle model
#         self._wheel_radius = 0.03
#         self._wheel_base = 0.1125
#         return

#     def forward(self, command):
#         # command will have two elements, first element is the forward velocity
#         # second element is the angular velocity (yaw only).
#         joint_velocities = [0.0, 0.0]
#         joint_velocities[0] = ((2 * command[0]) - (command[1] * self._wheel_base)) / (2 * self._wheel_radius)
#         joint_velocities[1] = ((2 * command[0]) + (command[1] * self._wheel_base)) / (2 * self._wheel_radius)
#         # A controller has to return an ArticulationAction
#         return ArticulationAction(joint_velocities=np.array(joint_velocities))

# def send_robot_actions(bot, controller, v, w):
#     position, orientation = bot.get_world_pose()
#     print(f"Bot's Position : {position}, Bot's Orientation : {orientation} ")
#     #apply the actions calculated by the controller
#     bot.apply_action(controller.forward(command=[v, w]))
#     return

#  # CONTROLLER CONTROL
# def go_forward(bot, controller, n_steps=50):
#     for _ in range(n_steps):
#         bot.apply_wheel_actions(controller.forward(np.array([1, 0])))
#     return None

# def go_backward(bot, controller, n_steps=50):
#     for _ in range(n_steps):
#         bot.apply_wheel_actions(controller.forward(np.array([-1, 0])))
#     return None

# def turn_right(bot, controller, n_steps=210):
#     for _ in range(n_steps):
#         bot.apply_wheel_actions(controller.forward(np.array([0, 0.5])))
#     return None

# def turn_left(bot, controller, n_steps=210):
#     for _ in range(n_steps):
#         bot.apply_wheel_actions(controller.forward(np.array([0, -0.5])))
#     return None

# def stay(bot, controller, n_steps=50):
#     for _ in range(n_steps):
#         bot.apply_wheel_actions(controller.forward(np.array([0, 0])))
#     return None

# def bot_act(bot, controller, action, n_steps=50):
#     if action == 0:
#         stay(bot, controller, n_steps)
#     elif action == 1:
#         go_forward(bot, controller, n_steps)
#     elif action == 2:
#         go_backward(bot, controller, n_steps)
#     elif action == 3:
#         turn_right(bot, controller, n_steps=210)
#     elif action == 4:
#         turn_left(bot, controller, n_steps=210)
    # return None

def bot_act(bot, controller, action):
    pos, ori = bot.get_world_pose()
    print(f"Bot's Position : {pos}, Bot's Orientation : {ori} ")
    if action == 1:
        bot.apply_action(controller.forward(start_position=pos, start_orientation=ori, goal_position=pos+np.array([0.0, 0.5, 0])))
    elif action == 2:
        bot.apply_action(controller.forward(start_position=pos, start_orientation=ori, goal_position=pos-np.array([0.0, 0.5, 0])))
    elif action == 3:
        bot.apply_action(controller.forward(start_position=pos, start_orientation=ori, goal_position=pos+np.array([0.5, 0.0, 0])))
    elif action == 4:
        bot.apply_action(controller.forward(start_position=pos, start_orientation=ori, goal_position=pos-np.array([0.5, 0.0, 0])))
    elif action == 0:
        bot.apply_action(controller.forward(start_position=pos, start_orientation=ori, goal_position=pos))
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
    # wr_stage = omni.usd.get_context().get_stage()

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

# scene = Scene()
# world.scene.add(XFormPrimView(prim_paths_expr="/World/Waiting_Room_Base"))

wr_world = World()
print("Waiting Room in the World!")

enable_extension("omni.isaac.debug_draw")
lidar_config = 'RPLIDAR_S2E'
wr_timeline = omni.timeline.get_timeline_interface() 
wr_world.initialize_physics()

# # JETBOT
# jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
# wr_world.scene.add(
#     WheeledRobot(
#         prim_path="/World/Jetbot",
#         name="Jetbot",
#         wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
#         create_robot=True,
#         usd_path=jetbot_asset_path,
#         position=np.array([0.4, -0.4,0]),
#     )
# )
# jetbot = ArticulationView(prim_paths_expr="/World/Jetbot" , name="Jetbot")
# print("Jetbot in the World")
# jetbot = wr_world.scene.get_object("Jetbot")
# jetbot.set_local_scale(np.array([2.5, 2.5, 2.5]))

# # JETBOT CAMERA
# jet_camera = Camera(prim_path='/World/Jetbot/chassis/rgb_camera/jetbot_camera', 
#                     name='Jetbot_Camera',
#                     frequency=30,
#                     resolution=(512,512),
#                     )
# jet_camera.initialize()
# kit.update()
# jet_camera.initialize()
# wr_world.initialize_physics()

# # JETBOT LiDAR
# _, sensor = omni.kit.commands.execute(
#     "IsaacSensorCreateRtxLidar",
#     path="/sensor",
#     parent=None,
#     config=lidar_config,
#     translation=(0, 0, 1.0),
#     orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # Gf.Quatd is w,i,j,k
# )
# hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")

# simulation_context = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
# kit.update()

# rp = rep.create.render_product(jet_camera, (512,512))
# jet_camera.add_bounding_box_2d_tight_to_frame()
# bbox_2d_tight = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
# bbox_2d_tight.attach(rp)

# print('Bot Created!')

# # my_controller = OpenController()
# bot_controller=DifferentialController(name="simple_control", wheel_radius=0.035, wheel_base=0.1125)
# throttle = 1.0
# steering = 0.5
# print("Controller Created!")

# val = 200
# turn = 80
# tmp = np.array([np.full(shape=val, fill_value=1), np.full(shape=val, fill_value=2), np.full(shape=val, fill_value=0), 
#                np.full(shape=val, fill_value=4), np.full(shape=val, fill_value=1), np.full(shape=val, fill_value=3), np.full(shape=val, fill_value=1)])

# i = 0
# jet_camera.add_motion_vectors_to_frame()

# while kit.is_running():
#     wr_world.step(render=True)
#     if i % 100 == 0:
#         act_val = np.random.randint(0,5,1)
#         bot_act(jetbot, bot_controller, act_val)
#         print(jet_camera.get_current_frame())    
#         imgplot = plt.imshow(jet_camera.get_rgba()[:, :, :3])
#         plt.show()
#         print(jet_camera.get_current_frame()["motion_vectors"])
#     if wr_world.is_playing():
#         if wr_world.current_time_step_index == 0:
#             wr_world.reset()
#     i += 1

# LEATHERBACK
leatherback_asset_path = assets_root_path + "/Isaac/Robots/Leatherback/leatherback.usd"
# wr_world.scene.add(
#     WheeledRobot(
#         prim_path="/World/Leatherback",
#         name="Leatherback",
#         wheel_dof_names=["Wheel__Upright__Rear_Right", "Wheel__Upright__Rear_Left", "Wheel__Knuckle__Front_Right", "Wheel__Knuckle__Front_Left"],
#         wheel_dof_indices=[13, 16, 24, 25],
#         create_robot=True,
#         usd_path=leatherback_asset_path,
#         position=np.array([0.4, -0.4, 0]),
#     )
# )

wr_world.scene.add(
    WheeledRobot(
        prim_path="/World/Leatherback",
        name="Leatherback",
        wheel_dof_names=["Wheel__Knuckle__Front_Left", "Wheel__Knuckle__Front_Right"],
        wheel_dof_indices=[24, 25],
        create_robot=True,
        usd_path=leatherback_asset_path,
        position=np.array([0.4, -0.4,0]),
    )
)
leatherback = wr_world.scene.get_object("Leatherback")
leatherback = ArticulationView(prim_paths_expr="/World/Leatherback" , name="Leatherback")
print("Leatherback in the World")

# LEATHERBACK CAMERA
lea_camera = Camera(prim_path="/World/Leatherback/Rigid_Bodies/Chassis/Camera_Right", 
                    name='Leatherback_Camera',
                    frequency=30,
                    resolution=(512,512),
                    )
lea_camera.initialize()
kit.update()
lea_camera.initialize()
wr_world.initialize_physics()

# LEATHERBACK LiDAR

# LIDAR RTX
lea_lidar = wr_world.scene.add(
    LidarRtx(prim_path="/World/Leatherback/Rigid_Bodies/Chassis/Lidar", 
             name="Leatherback_Lidar"))
print("performing reset")
wr_world.reset()
print("reset done")
lea_lidar.add_range_data_to_frame()
lea_lidar.add_point_cloud_data_to_frame()
lea_lidar.enable_visualization()

# my_controller = OpenController()
# bot_controller=DifferentialController(name="simple_control", wheel_radius=0.035, wheel_base=0.1125)
lea_controller = WheelBasePoseController(name='lea_controller', 
                                         open_loop_wheel_controller=DifferentialController(name='lea_diff_controller', 
                                                                                           wheel_radius=0.0995, wheel_base=0.21),
                                        is_holonomic=False)
print("Controller Created!")
print(f"Total DOF : {leatherback.num_dof}")
print(f"Bot's DOF Names : {leatherback.dof_names}")
print(f"Bot's DOF  Properties : {leatherback.dof_properties}")
print(f"Bot's Names Dtype {leatherback.dof_properties.dtype.names}")

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
                f.write(str(lea_lidar.get_current_frame()))
                f.write('\n')
            if i == 100:
                act_val = np.random.randint(0,5,1)
                act_val = np.array([2])
                print(f"Action Value : {act_val} - {act_dict[act_val[0]]}")
                bot_act(leatherback, lea_controller, act_val)
                with open('/home/rah_m/Isaac_World_Files/camera_data.txt', 'a+') as f:
                    f.write(str(lea_camera.get_current_frame()))
                    f.write('\n')
                # print(lea_camera.get_current_frame())    
                imgplot = plt.imshow(lea_camera.get_rgba()[:, :, :3])
                plt.show()
            if i == 1000:
                act_val = np.random.randint(0,5,1)
                act_val = np.array([1])
                print(f"Action Value : {act_val} - {act_dict[act_val[0]]}")
                bot_act(leatherback, act_val)
                with open('/home/rah_m/Isaac_World_Files/camera_data.txt', 'a+') as f:
                    f.write(str(lea_camera.get_current_frame()))
                    f.write('\n')
                # print(lea_camera.get_current_frame())    
                imgplot = plt.imshow(lea_camera.get_rgba()[:, :, :3])
                plt.show()
                    
                # print(lea_camera.get_current_frame()["motion_vectors"])
            # print(lea_lidar.get_current_frame())
            # forward
            # leatherback.apply_wheel_actions(bot_controller.forward(command=[0.0, np.pi/2]))
        else:
            break
        # elif i >= 1000 and i < 1265:
        #     # rotate
        #     leatherback.apply_wheel_actions(bot_controller.forward(command=[0.0, np.pi / 12]))
        # elif i >= 1265 and i < 2000:
        #     # forward
        #     leatherback.apply_wheel_actions(bot_controller.forward(command=[0.05, 0]))
        # elif i == 2000:
        #     i = 0
        i += 1
wr_world.stop()

# i = 0
# while kit.is_running():
#     wr_world.step(render=True)
#     if i % 100 == 0:
#         act_val = np.random.randint(0,5,1)
#         bot_act(leatherback, bot_controller, act_val)
#         print(lea_camera.get_current_frame())    
#         imgplot = plt.imshow(lea_camera.get_rgba()[:, :, :3])
#         plt.show()
#         print(lea_camera.get_current_frame()["motion_vectors"])
#     if wr_world.is_playing():
#         if wr_world.current_time_step_index == 0:
#             wr_world.reset()
#     i += 1

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

