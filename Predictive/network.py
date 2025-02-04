import torch
import torch.nn as nn

class NavigationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.control_net = nn.Sequential(
            nn.Linear(6, 16),  # Input: [bot_x, bot_y, goal_x, goal_y, world_theta, relative_theta]
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 2),  # Output: [L, delta_theta]
            nn.Tanh()  # Tanh activation for bounded output
        )
        
    def forward(self, current_state):
        return self.control_net(current_state)