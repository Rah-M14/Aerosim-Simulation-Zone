import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class NavigationNet(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space['vector'].shape[0]

        self.features = nn.Sequential(
            nn.Linear(n_input_channels, 16),  # New input: [bot_x, bot_y, goal_x, goal_y, world_theta, relative_theta]
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.linear_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()  # L will be in [0,1]
        )
        self.angular_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()  # delta_theta will be in [0,1]
        )
        self.distance_temp = torch.tensor(2)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        vector: [6] (input features)
        lidar_mask: [360] (input features)
        """

        raw_output = self.features(observations['vector'])

        L = self.linear_head(raw_output)  # [0,1] magnitude
        theta_raw = (self.angular_head(raw_output) * (2 * torch.pi)) % (2 * torch.pi)  # [0, 2π]

        # print(f"L: {L.shape}, theta_raw: {theta_raw.shape}")

        theta_deg = torch.rad2deg(theta_raw)
        theta_bin = theta_deg.long()
        all_bins = torch.arange(360, device=observations['vector'].device)
        bin_distances = torch.abs(all_bins.float() - theta_bin)
        # print(f"all_bins: {all_bins.shape}, bin_distances: {bin_distances.shape}, theta_bin: {theta_bin.shape}")
        
        safe_weights = F.softmax(-bin_distances / self.distance_temp, dim=-1)
        safe_weights = safe_weights * observations['lidar_mask']  # Zero out unsafe
        safe_probs = F.gumbel_softmax(safe_weights.log(), tau=0.5, hard=True)
        # print(f"safe_weights: {safe_weights.shape}, safe_probs: {safe_probs.shape}")

        nearest_bin = torch.argmax(safe_probs, dim=-1)
        theta_safe_deg = (nearest_bin.float())
        theta_safe = torch.deg2rad(theta_safe_deg)
        # print(f"nearest_bin: {nearest_bin.shape}, theta_safe: {theta_safe.shape}")
    
        confidence = 1 / (1 + bin_distances.gather(-1, nearest_bin.unsqueeze(-1))).squeeze(-1)
        # print(f"confidence: {confidence.shape}")
        final_theta = confidence * theta_raw.squeeze(-1) + (1 - confidence) * theta_safe
        # print(f"final_theta: {final_theta.shape}")

        act = torch.stack([L.squeeze(-1), final_theta], dim=-1)

        # print(f"Act Predicted: {act}")

        return torch.cat((act, torch.tensor(observations['lidar_mask'], device=observations['vector'].device)), dim=-1)