from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.wheeled_robots.robots import WheeledRobot

import numpy as np

class RLBotController(BaseController):
    def __init__(self):
        super().__init__(name="rl_bot_controller")
        # Nova Carter's parameters
        self._wheel_radius = 0.41
        self._wheel_base = 0.413
        print("RLBotController initialized")
        return

    def forward(self, command):
        # command will have two elements, first element is the forward velocity & second element is the angular velocity (yaw only).
        joint_velocities = [0.0, 0.0, 0.0, 0.0]
        joint_velocities[0] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        joint_velocities[1] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        joint_velocities[2] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        joint_velocities[3] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        return ArticulationAction(joint_velocities=joint_velocities)
    
class RLBotAct():
    def __init__(self, bot, controller, n_steps=5):
        self.bot = bot
        self.controller = controller
        self.n_steps = n_steps
        print("RLBotAct initialized")
    
    def move_bot(self, vals: np.array):
        if vals.shape[0] != 2:
            raise ValueError("The input array should have two elements, first element is the forward velocity & second element is the angular velocity.")
        for _ in range(self.n_steps):
            self.bot.apply_wheel_actions(self.controller.forward(vals))
        return None

# DISCRETE CONTROLLER BACKUP
# # CONTROLLER CONTROL
# def go_forward(bot, controller, n_steps=5):
#     for _ in range(n_steps):
#         bot.apply_wheel_actions(controller.forward(np.array([1, 0])))
#     return None

# def go_backward(bot, controller, n_steps=5):
#     for _ in range(n_steps):
#         bot.apply_wheel_actions(controller.forward(np.array([-1, 0])))
#     return None

# def turn_right(bot, controller, n_steps=20):
#     for _ in range(n_steps):
#         bot.apply_wheel_actions(controller.forward(np.array([0, -np.pi/2])))
#     return None

# def turn_left(bot, controller, n_steps=20):
#     for _ in range(n_steps):
#         bot.apply_wheel_actions(controller.forward(np.array([0, np.pi/2])))
#     return None

# def stay(bot, controller, n_steps=5):
#     for _ in range(n_steps):
#         bot.apply_wheel_actions(controller.forward(np.array([0, 0])))
#     return None

# def bot_act(bot, controller, action, n_steps=5):
#     if action == 0:
#         stay(bot, controller, n_steps)
#     elif action == 1:
#         go_forward(bot, controller, n_steps)
#     elif action == 2:
#         go_backward(bot, controller, n_steps)
#     elif action == 3:
#         turn_right(bot, controller, n_steps)
#     elif action == 4:
#         turn_left(bot, controller, n_steps)
#     return None

# act_dict = {0 : 'stay', 1 : 'forward', 2 : 'backward', 3 : 'right', 4 : 'left'}