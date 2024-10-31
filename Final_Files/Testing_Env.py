import argparse
import os
import pickle
import torch as th
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from L_Theta_RL import SocEnv
# import logger
import wandb

def load_model_eval(algo, model_path, replay_buffer_path=None):
    if algo.lower() == 'ppo':
        model = PPO.load(model_path)
    elif algo.lower() == 'sac':
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    if replay_buffer_path and os.path.exists(replay_buffer_path):
        with open(replay_buffer_path, 'rb') as f:
            model.replay_buffer = pickle.load(f)
        print(f"Replay buffer loaded from {replay_buffer_path}")

    return model

def test_model(algo, model_path, replay_buffer_path, log_path, render, num_episodes=5):
    # Create the environment
    env = SocEnv(algo=algo, botname='jackal', headless=(not render), state_normalize=True, logdir='/tmp/test_logs', mlp_context=32, img_context=16)
    env = DummyVecEnv([lambda: env])  # Wrap in a DummyVecEnv for compatibility

    # Load the model and replay buffer
    model = load_model_eval(algo, model_path, replay_buffer_path)
    mean_reward, std_reward = evaluate_policy(model.policy, env, n_eval_episodes=5, deterministic=True)

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    # with open(log_path, 'w+') as f:
    #     f.write(f"mean_reward={mean_reward:.2f} +/- {std_reward} \n")

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if render:
                env.render()
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trained RL model with rendering")
    parser.add_argument('--algo', type=str, choices=['ppo', 'sac'], required=True, help='RL algorithm used for training')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--replay_buffer_path', type=str, help='Path to the replay buffer file (optional)')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to run for testing')
    parser.add_argument('--render', action='store_true', help='Render the environment during testing')
    parser.add_argument('--log_path', type=str, help='path to log rewards')
    args = parser.parse_args()

    wandb.init(project="Testing_Soc_Nav")

    print("Testing Now!")
    test_model(args.algo, args.model_path, args.replay_buffer_path, args.num_episodes, args.render, args.log_path)