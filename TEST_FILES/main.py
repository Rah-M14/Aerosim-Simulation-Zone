import os
import glob
import numpy as np
import wandb
import argparse
import torch
import torch.nn as nn

from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import safe_mean
from tqdm.auto import tqdm

from env import PathFollowingEnv
from configs import ObservationConfig
from feature_ex import SocialNavigationExtractor

obs_config = ObservationConfig()

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self):
        self.pbar.update(1)
        return True
    
    def _on_training_end(self):
        self.pbar.close()
        self.pbar = None

class WandbCallback(BaseCallback):
    def __init__(self, enable_wandb: bool = False, verbose=0):
        super().__init__(verbose)
        self.global_step = 0
        self.enable_wandb = enable_wandb

    def _on_step(self):
        self.global_step += 1
        # Log training metrics and Learning Curve
        if self.enable_wandb and (len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0):
            wandb.log({"train/global_step": self.global_step})
            wandb.log({"train/episode_reward": safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]), "train/global_step": self.global_step})
            wandb.log({"train/episode_length": safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]), "train/global_step": self.global_step})
        return True
    
class CkptAutoRemovalCallback(BaseCallback):
    def __init__(self, model_dir, algo, keep_num=3, frequency=100000):
        super().__init__()
        self.model_dir = model_dir
        self.algo = algo
        self.keep_num = keep_num
        self.frequency = frequency

    def _on_step(self):
        ckpts = os.listdir(self.model_dir)
        ckpts.sort(key=lambda x: os.path.getmtime(os.path.join(self.model_dir, x)))
        for ckpt in ckpts[:-self.keep_num]:
            os.remove(os.path.join(self.model_dir, ckpt))
        return True

def make_env(algo, enable_reward_monitor: bool = False, enable_wandb: bool = False, render_type: str = "normal"):
    env = PathFollowingEnv(
        image_path=r"TEST_FILES\New_WR_World.png",
        algo_run=algo,
        max_episode_steps=obs_config.max_episode_steps,
        chunk_size=obs_config.chunk_size,
        headless=False,
        enable_reward_monitor=enable_reward_monitor,
        enable_wandb=enable_wandb, 
        render_type=render_type
    )
    return env

def get_next_run_folder(base_path):
    os.makedirs(base_path, exist_ok=True)
    existing_runs = glob.glob(os.path.join(base_path, "run_*"))
    if not existing_runs:
        return "run_1"
    run_numbers = [int(run.split("_")[-1]) for run in existing_runs]
    return f"run_{max(run_numbers) + 1}"

def get_checkpoint_path(model_dir, algorithm, checkpoint_type="latest"):
    if checkpoint_type == "best":
        checkpoint_path = os.path.join(model_dir, "best_model.zip")
    else:  # latest
        checkpoints = glob.glob(os.path.join(model_dir, "checkpoint_*.zip"))
        if not checkpoints:
            return None
        checkpoint_path = max(checkpoints, key=os.path.getmtime)
    return checkpoint_path if os.path.exists(checkpoint_path) else None

def main(args):
    algorithm = args.algo.upper()
    given_ckpt = args.ckpt_path
    
    base_dir = r"logs"
    run_id = get_next_run_folder(os.path.join(base_dir, "logs", algorithm))
    name = f"{algorithm}_{run_id}"
    
    if args.wandb_log:
        run = wandb.init(
            project="path-following-rl",
            name=name,
            config={
                "algorithm": algorithm,
                "learning_rate": 3e-4,
                "batch_size": 256,
                "buffer_size": 1000000,
                "gamma": 0.99,
                "tau": 0.005,
                "train_freq": 1,
                "gradient_steps": 1,
                "learning_starts": 10000,
                "policy_architecture": [256, 512, 512, 256],
                "value_architecture": [256, 512, 512, 256],
                "total_timesteps": 10000000,
            }
        )
    # else:
        # wandb.init(project="path-following-rl", name=name)
    
    log_dir = os.path.join(base_dir, "logs", algorithm, run_id)
    model_dir = os.path.join(base_dir, "models", algorithm, run_id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = make_env(name, enable_reward_monitor=True, enable_wandb=args.wandb_log, render_type=args.render_type)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Create evaluation environment
    eval_env = make_env(name, enable_reward_monitor=True, enable_wandb=args.wandb_log, render_type=args.render_type)
    eval_env = Monitor(eval_env, log_dir)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[16, 32, 64, 32, 16],
            qf=[16, 32, 64, 32, 16]
        )
    )

    # policy_kwargs = dict(
    # features_extractor_class=SocialNavigationExtractor,
    # features_extractor_kwargs=dict(
    #     lidar_points=360,
    #     path_points=10,
    #     social_features=8
    # ),
    # net_arch=dict(pi=[32, 64, 128, 256, 128], qf=[32, 64, 128, 256, 128]),  # Policy network architecture
    # activation_fn=nn.ReLU
    # )

    # Initialize model
    if given_ckpt:
        checkpoint_path = given_ckpt
        print(f"Resuming training from Given Checkpoint: {checkpoint_path}")
    elif args.resume:
        checkpoint_path = get_checkpoint_path(os.path.join(base_dir, "models", algorithm), algorithm, args.resume)
        if checkpoint_path:
            print(f"Resuming training from checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = None
        print(f"No checkpoint found, starting new training")

    if checkpoint_path is not None:
        if algorithm == "SAC":
            model = SAC.load(checkpoint_path, env=env, tensorboard_log=log_dir)
        elif algorithm == "PPO":
            model = PPO.load(checkpoint_path, env=env, tensorboard_log=log_dir)
        elif algorithm == "TD3":
            model = TD3.load(checkpoint_path, env=env, tensorboard_log=log_dir)
    else:
        print(f"No checkpoint found, starting new training")
        if algorithm == "SAC":
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=1000000,
                batch_size=256,
                ent_coef='auto',
                gamma=0.99,
                tau=0.005,
                train_freq=1,
                gradient_steps=1,
                learning_starts=10000,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=log_dir
            )
        elif algorithm == "PPO":
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                verbose=1,
                n_steps=2048,
                n_epochs=10,
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0001,
                vf_coef=0.5,
                max_grad_norm=0.5,
                normalize_advantage=True,
                policy_kwargs=policy_kwargs,
                tensorboard_log=log_dir
            )
        elif algorithm == "TD3":
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.1 * np.ones(n_actions)
            )

            model = TD3(
                "MlpPolicy",
                env,
                learning_rate=0.001,
                buffer_size=1000000,
                learning_starts=100,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                action_noise=action_noise,
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                # optimize_memory_usage=True,
                tensorboard_log="./tensorboard_logs/td3/",
                verbose=1
            )

    original_action_net = model.policy.action_net
    model.policy.action_net = nn.Sequential(
        original_action_net,
        nn.Tanh()
    )
        
    # Print complete model architecture
    print("\n=== Complete Model Architecture ===")
    print("\n1. Policy Network (Actor):")
    print(f"Input Shape: {env.observation_space.shape}")
    print("\nMLP Extractor Policy Network:")
    for name, module in model.policy.mlp_extractor.policy_net.named_children():
        print(f"  {name}: {module}")
    
    print("\nAction Network (Final layer):")
    print(f"  Action Net: {model.policy.action_net}")
    print(f"Output Shape: {env.action_space.shape}")

    print("\n2. Value Network (Critic):")
    print("\nMLP Extractor Value Network:")
    for name, module in model.policy.mlp_extractor.value_net.named_children():
        print(f"  {name}: {module}")
    
    print("\nValue Network (Final layer):")
    print(f"  Value Net: {model.policy.value_net}")
    
    # Print parameter counts
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n=== Parameter Counts ===")
    print(f"Total Trainable Parameters: {count_parameters(model.policy):,}")
    print(f"Policy Network Parameters: {count_parameters(model.policy.mlp_extractor.policy_net) + count_parameters(model.policy.action_net):,}")
    print(f"Value Network Parameters: {count_parameters(model.policy.mlp_extractor.value_net) + count_parameters(model.policy.value_net):,}")

    # model_till_last = nn.Sequential(*list(model.policy.mlp_extractor.policy_net.children()))
    # print("\nModel architecture till last layer:")
    # for idx, layer in enumerate(model_till_last):
    #     print(f"Layer {idx}: {layer}")

    if args.policy:
        pretrained_policy = torch.load(args.policy)
        pretrained_policy['model_state_dict'] = {k.replace('_orig_mod.', ''): v for k, v in pretrained_policy['model_state_dict'].items()}
        print("Model Policy:")
        for key,_ in model.policy.state_dict().items():
            print(key)
        print("\n")

        print("Loaded Policy:")
        for key in pretrained_policy['model_state_dict']:
            print(key)
        print("\n")

        pretrained_policy['model_state_dict']['value_net.weight'] = model.policy.value_net.weight
        pretrained_policy['model_state_dict']['value_net.bias'] = model.policy.value_net.bias
        model.policy.load_state_dict(pretrained_policy['model_state_dict'])

    total_timesteps = 1000000000


    callbacks = [
        CheckpointCallback(
            save_freq=10000,
            save_path=model_dir,
            name_prefix=f"{algorithm}_checkpoint",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=5000,
            deterministic=True,
            render=False
        ),
        CkptAutoRemovalCallback(model_dir, algorithm, keep_num=3, frequency=50000),
        WandbCallback(enable_wandb=args.wandb_log),
        ProgressBarCallback(total_timesteps)
    ]

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10
        )

        final_model_path = f"{model_dir}/final_model_{algorithm}"
        model.save(final_model_path)
        env.save(f"{model_dir}/vec_normalize.pkl")
        
        wandb.save(f"{final_model_path}.zip")
        wandb.save(f"{model_dir}/vec_normalize.pkl")
        
        print("\nTraining completed successfully!")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        model.save(f"{model_dir}/interrupted_model")
        env.save(f"{model_dir}/vec_normalize.pkl")
        
    finally:
        wandb.finish()

def evaluate_model(args, num_episodes=100):

    algo = args.algo.upper()
    model_path = args.ckpt_path
    policy_path = args.policy

    wandb.init(
        project="path-following-eval",
        name="evaluation",
        config={"num_episodes": num_episodes}
    )
    
    env = make_env(algo, enable_reward_monitor=True, enable_wandb=args.wandb_log)
    env = DummyVecEnv([lambda: env])
    # env = VecNormalize.load(f"models/vec_normalize.pkl", env)

    policy_kwargs = {
                "net_arch": dict(
                    pi=[16, 32, 64, 32, 16],
                    qf=[16, 32, 64, 32, 16]
                ),
                "activation_fn": nn.ReLU
                # "optimizer_class": optim.AdamW,
                # "optimizer_kwargs": dict(weight_decay=1e-5)
            }

    if model_path and algo == "SAC":   
        model = SAC.load(model_path)
    elif model_path and algo == "PPO":
        model = PPO.load(model_path)
    elif model_path and algo == "TD3":
        model = TD3.load(model_path)

    elif algo == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1000000,
            batch_size=256,
            ent_coef='auto',
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            learning_starts=10000,
            policy_kwargs=policy_kwargs,
            verbose=1,
            )
    elif algo == "PPO":
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                verbose=1,
                n_steps=2048,
                n_epochs=10,
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                normalize_advantage=True,
                policy_kwargs=policy_kwargs,
            )
    elif algo == "TD3":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
            )

        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=0.001,
            buffer_size=1000000,
            learning_starts=100,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            action_noise=action_noise,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            # optimize_memory_usage=True,
            verbose=1
            )
        
    original_action_net = model.policy.action_net
    model.policy.action_net = nn.Sequential(
        original_action_net,
        nn.Tanh()
    )
        
    # Print complete model architecture
    print("\n=== Complete Model Architecture ===")
    print("\n1. Policy Network (Actor):")
    print(f"Input Shape: {env.observation_space.shape}")
    print("\nMLP Extractor Policy Network:")
    for name, module in model.policy.mlp_extractor.policy_net.named_children():
        print(f"  {name}: {module}")
    
    print("\nAction Network (Final layer):")
    print(f"  Action Net: {model.policy.action_net}")
    print(f"Output Shape: {env.action_space.shape}")

    print("\n2. Value Network (Critic):")
    print("\nMLP Extractor Value Network:")
    for name, module in model.policy.mlp_extractor.value_net.named_children():
        print(f"  {name}: {module}")
    
    print("\nValue Network (Final layer):")
    print(f"  Value Net: {model.policy.value_net}")
    
    # Print parameter counts
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n=== Parameter Counts ===")
    print(f"Total Trainable Parameters: {count_parameters(model.policy):,}")
    print(f"Policy Network Parameters: {count_parameters(model.policy.mlp_extractor.policy_net) + count_parameters(model.policy.action_net):,}")
    print(f"Value Network Parameters: {count_parameters(model.policy.mlp_extractor.value_net) + count_parameters(model.policy.value_net):,}")

    # model_till_last = nn.Sequential(*list(model.policy.mlp_extractor.policy_net.children()))
    # print("\nModel architecture till last layer:")
    # for idx, layer in enumerate(model_till_last):
    #     print(f"Layer {idx}: {layer}")

    if args.policy:
        pretrained_policy = torch.load(args.policy)
        pretrained_policy['model_state_dict'] = {k.replace('_orig_mod.', ''): v for k, v in pretrained_policy['model_state_dict'].items()}
        print("Model Policy:")
        for key,_ in model.policy.state_dict().items():
            print(key)
        print("\n")

        print("Loaded Policy:")
        for key in pretrained_policy['model_state_dict']:
            print(key)
        print("\n")

        pretrained_policy['model_state_dict']['value_net.weight'] = model.policy.value_net.weight
        pretrained_policy['model_state_dict']['value_net.bias'] = model.policy.value_net.bias
        model.policy.load_state_dict(pretrained_policy['model_state_dict'])

    rewards = []
    
    with tqdm(total=num_episodes, desc="Evaluating") as pbar:
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, val = model.predict(obs, deterministic=True)
                print(f"Observations: {obs}")
                print(f"Action: {action}")
                print(f"Value: {val}")
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                
                # Log evaluation metrics
                wandb.log({
                    "eval/step_reward": reward[0],
                    "eval/distance_to_goal": info[0].get('distance_to_goal', 0),
                    "eval/distance_to_next": info[0].get('distance_to_next', 0),
                    "eval/action_linear": info[0].get('action_linear', 0),
                    "eval/action_angular": info[0].get('action_angular', 0)
                })
                
            rewards.append(episode_reward)
            wandb.log({
                "eval/episode": episode,
                "eval/episode_reward": episode_reward,
                "eval/episode_length": info[0].get('episode_length', 0),
                "eval/success": info[0].get('success', False)
            })
            
            pbar.update(1)
            pbar.set_postfix({'reward': f'{episode_reward:.2f}'})
    
    wandb.log({
        "eval/average_reward": np.mean(rewards),
        "eval/reward_std": np.std(rewards)
    })
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained model")
    parser.add_argument("--wandb_log", action="store_true", help="Enable WandB logging")
    parser.add_argument("--render_type", type=str, default="normal", choices=["normal", "dual"], help="Render jsut on black bg (normal) or with the image (dual)")
    parser.add_argument("--policy", type=str, help="Pretrained base policy")
    parser.add_argument("--resume", type=str, default=None, choices=["latest", "best"], help="Resume the checkpoint")
    parser.add_argument("--algo", type=str, default="sac", choices=["ppo", "sac", "td3"], help="Algorithm to use")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the checkpoint")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_model(args)
    else:
        main(args)