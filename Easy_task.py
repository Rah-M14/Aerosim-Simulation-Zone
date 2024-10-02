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
# from omni.isaac.core.utils import stage
# from omni.isaac.core.scenes import Scene
# from omni.isaac.core.prims import XFormPrimView
from omni.isaac.nucleus import get_assets_root_path, is_file

import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.replicator.core as rep
import omni.timeline
from omni.isaac.core.utils.extensions import disable_extension, enable_extension
import time
import json

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

from RL_Bot import RLBot
from RL_Bot_Control import RLBotController, RLBotAct
from Mod_Pegasus_App import PegasusApp
# from Pegasus_App import PegasusApp

# THE WORLD SECTION STARTS HERE

def convert_to_serializable(data):
    """
    Convert numpy ndarrays to lists for JSON serialization.
    
    Args:
        data: The data to convert (can be a dict, list, or ndarray).
        
    Returns:
        The converted data.
    """
    if isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert ndarray to list
    else:
        return data  # Return as is for other types

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

wr_bot = RLBot(wr_world, kit, assets_root_path)
wr_bot.bot_reset()
wr_bot_controller = RLBotController()
wr_bot_act = RLBotAct(wr_bot.rl_bot, wr_bot_controller, n_steps=5)
print("Bot Created!")

# for i in range(2000):
#     kit.update()
# print("launching pegasus")

pg_app = PegasusApp(wr_world, wr_timeline, kit)
# pg_app.run()

i = 1
reset_needed = False
while kit.is_running():
    wr_timeline.play()
    wr_world.step(render=True)
    if wr_world.is_stopped() and not reset_needed:
        reset_needed = True
    if wr_world.is_playing():
        # if reset_needed:
        #     wr_world.reset()
        #     # bot_controller.reset()
        #     reset_needed = False
        if i % 100 != 0:
            # print(wr_bot.rl_bot_lidar.get_current_frame())
            # print(type(wr_bot.rl_bot_lidar.get_current_frame()))
            with open(f'/home/rah_m/Isaac_World_Files/lidar_data_{i}.json', 'w') as f:
                json.dump(convert_to_serializable(wr_bot.rl_bot_lidar.get_current_frame()), f)
                # f.write(str(wr_bot.rl_bot_lidar.get_current_frame()))
                # f.write('\n')
            if i % 150 == 0:
                # rand_val = np.random.uniform(-2,2,1)
                # rand_val = np.append(rand_val, np.random.uniform(-np.pi, np.pi, 1))
                # print(f"Applied Action Values : {rand_val}")
                # wr_bot_act.move_bot(vals=rand_val)
                with open('/home/rah_m/Isaac_World_Files/camera_data.json', 'a+') as f:
                    f.write(str(wr_bot.rl_bot_camera.get_current_frame()))
                    f.write('\n')
                imgplot = plt.imshow(wr_bot.rl_bot_camera.get_rgba()[:, :, :3])
                plt.show()
        else:
            # carb.log_warn("PegasusApp Simulation App is closing.")
            print("Simulation Done!")
            # wr_timeline.stop()
            # wr_world.stop()
            kit.update()
            wr_world.reset(True)
            kit.update()
            # kit.close()
            # break
        i += 1
# wr_world.stop()
print('loop is done!')

# omni.timeline.get_timeline_interface().play()
# Run in test mode, exit after a fixed number of steps
# if args.test is True:
#     for i in range(10):
#         # Run in realtime mode, we don't specify the step size
#         kit.update()
# else:
#     while kit.is_running():
#         # Run in realtime mode, we don't specify the step size
#         kit.update()

# omni.timeline.get_timeline_interface().stop()