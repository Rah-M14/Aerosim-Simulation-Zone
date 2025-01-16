from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

@dataclass
class SimulationConfig:
    """Simulation-specific configuration parameters"""
    physics_dt: float = 1/2
    rendering_dt: float = 1/2
    max_episode_length: int = 1000
    skip_frame: int = 1
    gpu_config: Dict = None
    headless: bool = False
    simulation_app_config: Dict = None

    def __post_init__(self):
        if self.simulation_app_config is None:
            self.simulation_app_config = {
                "headless": self.headless,
                "width": 1280,
                "height": 720,
                "sync_loads": True,
                "active_gpu": 0,
                "physics_gpu": 0,
                "multi_gpu": False,
                "renderer": "RayTracedLighting",
            }

@dataclass
class RobotConfig:
    """Robot-specific configuration parameters"""
    name: str = "jackal"
    length: float = 0.508
    RL_length: float = 2*length
    theta: float = np.pi/9
    max_velocity: float = 1.0
    max_angular_velocity: float = np.pi * 4
    wheel_radius: float = 0.098
    wheel_base: float = 0.262

    def __post_init__(self):
        # Update parameters based on robot type
        if self.name.lower() == "carter":
            self.wheel_radius = 0.24
            self.wheel_base = 0.54
        elif self.name.lower() == "novacarter":
            self.wheel_radius = 0.14
            self.wheel_base = 0.3452
        elif self.name.lower() == "jackal":
            self.wheel_radius = 0.098
            self.wheel_base = 0.262

@dataclass
class RRTStarConfig:
    """RRTStar configuration parameters"""
    max_iter: int = 500
    step_size: float = 0.66
    neighbor_radius: float = 1.0

@dataclass
class ObservationConfig:
    """Observation space configuration parameters"""
    use_path: bool = True
    chunk_size: int = 12
    base_dim: int = 11
    mlp_context_length: int = 32
    img_context_length: int = 16
    state_normalize: bool = True
    image_size: int = 64
    channels: int = 3
    vector_dim: int = (base_dim + 2*chunk_size) if use_path else base_dim

@dataclass
class RewardConfig:
    """Reward calculation configuration parameters"""
    # COLLISION PARAMETERS
    coll_dist_threshold: float = 1.0
    coll_exp_rew_scale: float = 11
    min_cluster_size: float = 2
    cluster_eps: float = 0.5
    angle_weight: float = 0.5

    # BOUNDARY THRESHOLDS
    x_limits: Tuple[float, float] = (-9,9)
    y_limits: Tuple[float, float] = (-7,7)
    boundary_threshold: float = 0.5
    boundary_coll_penalty: float = -50

    # GOAL & TERMINAL REWARDS & THRESHOLDS
    timeout_penalty: float = -100
    close_goal_dist_threshold: float = 0.5
    goal_reward: float = 10000.0
    live_reward: float = -2.0

    # RRT* PATH REWARD PARAMS
    path_corridor_width: float = 0.5  # Acceptable deviation width
    max_path_deviation: float = 1.0   # Maximum deviation before heavy penalties
    path_following_factor: float = 5
    deviation_penalty_factor: float = -5
    
    # NEW SCALING PARAMETERS
    max_distance: float = 10.0  # Maximum possible distance in the environment
    boundary_limits: Tuple[float, float, float, float] = (-9.5, 9.5, -6.5, 6.5)  # (x_min, x_max, y_min, y_max)
    direction_scale_factors: Tuple[float,float] = (5, 10)
    distance_scale_factor: float = 100
    smooth_crash_threshold: float = 1.5

@dataclass
class TrainingConfig:
    """Training algorithm configuration parameters"""
    wandb_project_name: str = "Nav_Testing"
    algorithm: str = "PPO"
    ppo_config: Dict = None
    sac_config: Dict = None

    cnn_output_dim: int = 256

    def __post_init__(self):
        if self.ppo_config is None:
            self.ppo_config = {
                "n_steps": 2048,
                "batch_size": 256,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "ent_coef": 0.1,
                "clip_range": 0.3,
                "n_epochs": 20,
                "gae_lambda": 1.0,
                "max_grad_norm": 0.9,
                "vf_coef": 0.95,
                "use_sde": False,
            }
        if self.sac_config is None:
            self.sac_config = {
                "buffer_size": 10000,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "batch_size": 512,
                "tau": 0.005,
                "ent_coef": "auto_0.1",
                "train_freq": (1, "episode"),
                "gradient_steps": -1,
                "learning_starts": 5000,
                "use_sde": False,
                "sde_sample_freq": -1,
                "use_sde_at_warmup": False,
                "optimize_memory_usage": False,
            }

@dataclass
class EnvironmentConfig:
    """Main configuration class that combines all sub-configs"""
    simulation: SimulationConfig = SimulationConfig()
    robot: RobotConfig = RobotConfig()
    observation: ObservationConfig = ObservationConfig()
    reward: RewardConfig = RewardConfig()
    training: TrainingConfig = TrainingConfig()
