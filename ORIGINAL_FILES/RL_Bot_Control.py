from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.types import ArticulationAction

import numpy as np

class RLBot_NovaCarter_Controller(BaseController):
    def __init__(self):
        super().__init__(name="rl_bot_controller")
        # Nova Carter's parameters
        self._wheel_radius = 0.14
        self._wheel_base = 0.3452
        print("RLBotController initialized")
        return

    def forward(self, command):
        joint_velocities = [0.0, 0.0, 0.0, 0.0]

        joint_velocities[0] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        joint_velocities[1] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        joint_velocities[2] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        joint_velocities[3] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        return ArticulationAction(joint_velocities=joint_velocities)

class RLBot_Carter_Controller(BaseController):
    def __init__(self):
        super().__init__(name="rl_bot_controller")
        # Carter's parameters
        self._wheel_radius = 0.24
        self._wheel_base = 0.54
        print("RLBotController initialized")
        return

    def forward(self, command):
        joint_velocities = [0.0, 0.0]

        joint_velocities[0] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        joint_velocities[1] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        return ArticulationAction(joint_velocities=joint_velocities)

class RLBotController(BaseController):
    def __init__(self):
        super().__init__(name="rl_bot_controller")
        # Jackal's parameters
        self._wheel_radius = 0.098
        self._wheel_base = 0.262
        print("RLBotController initialized")
        return

    def forward(self, command):
        joint_velocities = [0.0, 0.0, 0.0, 0.0]

        joint_velocities[0] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        joint_velocities[1] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        joint_velocities[2] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        joint_velocities[3] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
        return ArticulationAction(joint_velocities=joint_velocities)

class RLBotAct():
    def __init__(self, bot, controller, n_steps=6):
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