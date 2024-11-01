from isaacsim import SimulationApp
kit = SimulationApp({"headless": False}) # we can also run as headless.

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
import numpy as np
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.controllers import BaseController
from omni.isaac.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
import numpy as np
import time

from New_RL_Bot import RLBot
from New_RL_Bot_Control import RLBotController, RLBotAct

# class RLBotController(BaseController):
#     def __init__(self, botname):
#         super().__init__(name="rl_bot_controller")
#         self.botname = botname
#         if self.botname.lower() == 'jackal':
#             # Jackal's parameters
#             self._wheel_radius = 0.098
#             self._wheel_base = 0.262
#             print("Jackal RLBotController initialized")

#         elif self.botname.lower() == 'carter':
#             # Carter's parameters
#             self._wheel_radius = 0.24
#             self._wheel_base = 0.54
#             print("Carter RLBotController initialized")
        
#         return

#     def forward(self, command):
#         if self.botname.lower() == 'jackal':
#             joint_velocities = [0.0, 0.0, 0.0, 0.0]

#             joint_velocities[0] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
#             joint_velocities[1] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
#             joint_velocities[2] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
#             joint_velocities[3] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
#             return ArticulationAction(joint_velocities=joint_velocities)

#         elif self.botname.lower() == 'carter':
#             joint_velocities = [0.0, 0.0]

#             joint_velocities[0] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
#             joint_velocities[1] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
#             return ArticulationAction(joint_velocities=joint_velocities)

def send_robot_actions(step_size, bot, goal_pos):
    position, orientation = bot.get_world_pose()
    bot.apply_action(my_controller.forward(start_position=position,
                                                        start_orientation=orientation,
                                                        goal_position=goal_pos,
                                                        lateral_velocity=0.5,
                                                        yaw_velocity=1.5,
                                                        heading_tol=0.05,
                                                        position_tol=0.1))
    return

def transform_coord(l,th):
    return np.array([np.cos(th)*l, np.sin(th)*l, 0.0])

world = World()
world.scene.add_default_ground_plane()
arp = get_assets_root_path()

# jetbot_asset_path = arp + "/Isaac/Robots/Jetbot/jetbot.usd"
# world.scene.add(WheeledRobot(
#                 prim_path="/World/Fancy_Robot",
#                 name="fancy_robot",
#                 wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
#                 create_robot=True,
#                 usd_path=jetbot_asset_path))

# jetbot = ArticulationView(prim_paths_expr="/World/Fancy_Robot", name="jetbot")

# world.scene.add(jetbot)
# world.initialize_physics()
# my_controller = WheelBasePoseController(name="cool_controller",
#                                                     open_loop_wheel_controller=
#                                                         DifferentialController(name="simple_control",
#                                                                                 wheel_radius=0.03, wheel_base=0.1125),
#                                                 is_holonomic=False)


bot = RLBot(simulation_app=kit, world=world, botname='jackal', assets_root_path=arp)
internal_controller = RLBotController(botname='jackal')
my_controller = WheelBasePoseController(name="cool_controller",
                                                    open_loop_wheel_controller=internal_controller,
                                                    is_holonomic=False)
bot_act = RLBotAct(kit, bot.rl_bot, my_controller)

world.reset()

dist_tol = 0.2
goal_pos = np.array([10, 10, 0])
len_ranges = [-0.1, 0.5]
theta_ranges = [-(np.pi)/4, (np.pi)/4]

goal_coords, new_pos = None, None
while True:
    # start_pos, start_ori = jetbot.get_world_poses()
    # print(f"start_pos : {start_pos}, start_ori : {start_ori}")
    # send_robot_actions(step_size=1, bot=jetbot)
    # new_pos, start_ori = jetbot.get_world_poses()
    # print(f"New_pos : {new_pos[0]}, start_ori : {start_ori[0]}")
    # print(f"to be at : [1.8, 1.8]")

    current_pos, current_ori = bot.rl_bot.get_world_pose()
    print(f"Current time step : {i}")
    print(f"start_pos : {current_pos}, start_ori : {current_ori}")


    if goal_coords is None or np.linalg.norm(new_pos[:2] - goal_coords[:2]) < dist_tol:
        goal_dist_x = goal_pos[0] - current_pos[0]
        goal_dist_y = goal_pos[1] - current_pos[1]
        chosen_l = np.random.uniform(low=len_ranges[0], high=len_ranges[1], size=1)
        # chosen_l = min(goal_dist_x, len_ranges[1])
        # chosen_ly = min(goal_dist_y, len_ranges[1])
        chosen_th = np.random.uniform(low=-(np.pi/4), high=(np.pi/4), size=1)
        # chosen_th = theta_ranges[1]
        coord = transform_coord(chosen_l[0], chosen_th[0])
        # goal_coords = np.array([x, y])
        goal_coords = current_pos + coord
        print(f"Goal to reach : {goal_coords}")

    bot_act.send_actions(start_pos=current_pos, start_ori=current_ori, goal_pos=goal_coords)
    new_pos, new_ori = bot.rl_bot.get_world_pose()

    print(f"New_pos : {new_pos}, start_ori : {new_ori}")
    print(f"Goal to reach : {goal_coords}")
    world.step(render=True) # execute one physics step and one rendering step

kit.close() # close Isaac Sim