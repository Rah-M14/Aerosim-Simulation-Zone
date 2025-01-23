import torch
import torch.nn as nn

class SocialNavigationExtractor(nn.Module):
    def __init__(self, lidar_points=360, path_points=10, social_features=8):
        super().__init__()
        
        # LiDAR Processing Branch (Collision Awareness)
        self.lidar_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(16)
        )
        
        # Path Following Branch
        self.path_encoder = nn.Sequential(
            nn.Linear(path_points * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Goal and Progress Branch
        self.goal_encoder = nn.Sequential(
            nn.Linear(4, 32),  # Current pos + goal pos
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Attention for Path-LiDAR Integration
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            batch_first=True
        )
        
        # Final Integration Layer
        self.integration = nn.Sequential(
            nn.Linear(64 + 64 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.layernorm = nn.LayerNorm(64)
        
    def forward(self, observations):
        lidar = observations[-1].unsqueeze(1)  # [_, 1, 360]
        path = observations[4:28]  # [_, P*2]
        current_pos = observations[:2]  # [_, 2]
        goal_pos = observations[2:4]  # [_, 2]
        
        # Process LiDAR data
        lidar_features = self.lidar_encoder(lidar)  # [_, 64, 16]
        lidar_features = lidar_features.transpose(1, 2)  # [_, 16, 64]
        
        # Process path data
        path_features = self.path_encoder(path)  # [_, 64]
        path_features = path_features.unsqueeze(1)  # [_, 1, 64]
        
        # Cross-attention between path and LiDAR
        path_aware_features, _ = self.cross_attention(
            path_features,
            lidar_features,
            lidar_features
        )
        path_aware_features = self.layernorm(path_aware_features.squeeze(1))
        
        # Process goal and current position
        goal_features = self.goal_encoder(
            torch.cat([current_pos, goal_pos], dim=-1)
        )
        
        # Integrate all features
        combined_features = torch.cat([
            path_aware_features,
            lidar_features.mean(dim=1),  # Global LiDAR context
            goal_features
        ], dim=-1)
        
        return self.integration(combined_features)
