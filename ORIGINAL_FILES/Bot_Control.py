from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.types import ArticulationAction

class RLBotController(BaseController):
    def __init__(self):
        super().__init__(name="rl_bot_controller")
        # In our case - Nova Carter
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

