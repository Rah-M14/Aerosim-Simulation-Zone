from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th

from Soc_Env import SocEnv

log_dir = "home/rah_m/Soc_Train_Logs"
my_env = SocEnv(headless=True)

policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[16, dict(pi=[128, 128, 128], vf=[128, 128, 128])])
policy = CnnPolicy
total_timesteps = 50000000

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="Soc_Bot_checkpoint")
model = PPO(
    policy,
    my_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=2560,
    batch_size=64,
    learning_rate=0.000125,
    gamma=0.9,
    ent_coef=7.5e-08,
    clip_range=0.3,
    n_epochs=5,
    device="cuda",
    gae_lambda=1.0,
    max_grad_norm=0.9,
    vf_coef=0.95,
    tensorboard_log=log_dir,
)

model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])
model.save(log_dir + "/Soc_Nav_Policy")
my_env.close()