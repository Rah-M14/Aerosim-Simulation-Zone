from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.types import ArticulationAction

import numpy as np

class RLBotController(BaseController):
    def __init__(self, botname):
        super().__init__(name="rl_bot_controller")
        self.botname = botname
        if self.botname.lower() == 'jackal':
            # Jackal's parameters
            self._wheel_radius = 0.098
            self._wheel_base = 0.262
            print("Jackal RLBotController initialized")

        elif self.botname.lower() == 'carter':
            # Carter's parameters
            self._wheel_radius = 0.24
            self._wheel_base = 0.54
            print("Carter RLBotController initialized")
        
        elif self.botname.lower() == 'novacarter':
            # Nova Carter's parameters
            self._wheel_radius = 0.14
            self._wheel_base = 0.3452
            print("Nova Carter RLBotController initialized")
        return

    def forward(self, command):
        if self.botname.lower() == 'jackal':
            joint_velocities = [0.0, 0.0, 0.0, 0.0]

            joint_velocities[0] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
            joint_velocities[1] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
            joint_velocities[2] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
            joint_velocities[3] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
            return ArticulationAction(joint_velocities=joint_velocities)

        elif self.botname.lower() == 'carter':
            joint_velocities = [0.0, 0.0]

            joint_velocities[0] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
            joint_velocities[1] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
            return ArticulationAction(joint_velocities=joint_velocities)
        
        elif self.botname.lower() == 'novacarter':
            joint_velocities = [0.0, 0.0, 0.0, 0.0]

            joint_velocities[0] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
            joint_velocities[1] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
            joint_velocities[2] = (((2 * command[0])/(self._wheel_radius)) - ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
            joint_velocities[3] = (((2 * command[0])/(self._wheel_radius)) + ((command[1] * self._wheel_base))/(2 * self._wheel_radius))
            return ArticulationAction(joint_velocities=joint_velocities)

class RLBotAct():
    def __init__(self, bot, controller, n_steps=10):
        self.bot = bot
        self.controller = controller
        self.n_steps = n_steps
        print("RLBotAct initialized")
    
    def move_bot(self, cur_pos, cur_ori, next_pos):
        self.bot.apply_wheel_actions(self.controller.forward(start_position=cur_pos, start_orientation=cur_ori, goal_position=next_pos, position_tol=0.1, heading_tol=0.2))
        return None