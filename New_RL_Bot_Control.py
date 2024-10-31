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
    def __init__(self, kit, bot, controller, n_steps=5):
        self.kit = kit
        self.bot = bot
        self.controller = controller
        self.n_steps = n_steps
        print("RLBotAct initialized")
    
    def move_bot(self, vals: np.array, n_steps: int = 5):
        if vals.shape[0] != 2:
            raise ValueError("The input array should have two elements, first element is the forward velocity & second element is the angular velocity.")
        self.bot.apply_wheel_actions(self.controller.forward(vals))
        # for _ in range(n_steps):
        #     self.kit.update()
        return None
    
    def stop_bot(self):
        self.bot.apply_wheel_actions(self.controller.forward(np.array([0.0, 0.0])))
        return None
    
    def send_actions(self, start_pos, start_ori, goal_pos):
        self.bot.apply_action(self.controller.forward(start_position=start_pos,
                                                        start_orientation=start_ori,
                                                        goal_position=goal_pos,
                                                        lateral_velocity=0.2,
                                                        yaw_velocity=2.5,
                                                        heading_tol=5,
                                                        position_tol=0.1))
        return None

import math

import numpy as np
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from omni.isaac.core.utils.types import ArticulationAction


class CustomWheelBasePoseController(BaseController):
    def __init__(self, name: str, open_loop_wheel_controller: BaseController, is_holonomic: bool = False) -> None:
        super().__init__(name)
        self._open_loop_wheel_controller = open_loop_wheel_controller
        self._is_holonomic = is_holonomic
        self.goal_reached = False
        # self.i = 0
        return

    def forward(
        self,
        start_position: np.ndarray,
        start_orientation: np.ndarray,
        goal_position: np.ndarray,
        lateral_velocity: float = 1.0,
        yaw_velocity: float = 2.5,
        heading_tol: float = 5,
        position_tol: float = 0.04,
    ) -> ArticulationAction:
        self.goal_reached = False
        # self.i += 1
        # print(f"Forward is called")
        
        _, _, current_yaw = np.rad2deg(quat_to_euler_angles(start_orientation))

        # Calculate desired heading (degrees)
        delta_x = goal_position[0] - start_position[0]
        delta_y = goal_position[1] - start_position[1]
        desired_heading = np.rad2deg(np.arctan2(delta_y, delta_x))

        # Calculate angle error (degrees)
        angle_error = (desired_heading - current_yaw + 180) % 360 - 180

        # Calculate distance to goal
        dist_to_goal = np.linalg.norm(start_position[:2] - goal_position[:2])

        # Determine command based on angle error and distance
        if dist_to_goal < position_tol:
            command = [0.0, 0.0]  # Stop if within position tolerance
            self.goal_reached = True
        elif dist_to_goal < position_tol*5:
            command = [lateral_velocity/2, 0]
        elif np.abs(angle_error) > heading_tol:
            command = [0.0, yaw_velocity if angle_error > 0 else -yaw_velocity]
        else:
            command = [lateral_velocity, 0.0]

        return self._open_loop_wheel_controller.forward(command)
    
    def reset(self) -> None:
        """[summary]"""
        return