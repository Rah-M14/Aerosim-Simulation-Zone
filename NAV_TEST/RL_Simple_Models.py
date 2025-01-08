import json
import logging
import glob
import os
import pickle
import sys

import torch as th
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo import MultiInputPolicy as PPOMultiPolicy
from stable_baselines3.sac import MultiInputPolicy as SACMultiPolicy

th.cuda.empty_cache()
th.backends.cudnn.benchmark = True

from configs import EnvironmentConfig

import gym
import numpy as np
from gym import spaces

env_config = EnvironmentConfig()

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_output_dim = env_config.training.cnn_output_dim,
        algo="ppo",
        mlp_context=env_config.observation.mlp_context_length,
        img_context=env_config.observation.img_context_length,
    ):
        super(CustomCombinedExtractor, self).__init__(
            observation_space, features_dim=2
        )  # Placeholder Feature dims for PyTorch call
        import torch.nn as nn

        self._observation_space = observation_space
        self.mlp_context = mlp_context
        self.img_context = img_context
        self.img_channels = env_config.observation.channels
        self.vec_lstm_out_size = 32
        self.img_lstm_out_size = 128
        self.n_flatten_size = 256

        # self.device = device

        self.vec_lstm = nn.LSTM(env_config.observation.vector_dim, self.vec_lstm_out_size, num_layers=1)
        self.img_cnn = nn.Sequential(
            nn.Conv2d(self.img_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.img_lstm = nn.LSTM(
            self.n_flatten_size, self.img_lstm_out_size, num_layers=2
        )

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if algo == "ppo":
                if key == "vector":
                    total_concat_size += self.vec_lstm_out_size
                elif key == "image":
                    total_concat_size += self.img_lstm_out_size

            elif algo == "sac":
                if key == "vector":
                    total_concat_size += self.vec_lstm_out_size
                elif key == "image":
                    total_concat_size += self.img_lstm_out_size

        self._features_dim = total_concat_size

    def get_device(self, network):
        return next(network.parameters()).device

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        model_device = self.get_device(self.img_lstm)

        for key, subspace in self._observation_space.spaces.items():
            if key == "vector":
                if observations[key].shape[0] == 1:
                    core_in = observations[key].squeeze(0)
                    # print("vec device nb", core_in.device)
                    lstm_out = self.vec_lstm(core_in)[-1][0]
                    encoded_tensor_list.append(lstm_out[-1].unsqueeze(0))
                else:
                    core_in = (
                        observations[key].view(self.mlp_context, -1, 9).to(model_device)
                    )
                    # print("vec device batch", core_in.device)
                    lstm_out = self.vec_lstm(core_in)[1][0]
                    encoded_tensor_list.append(lstm_out[-1])

            else:
                if observations[key].shape[0] == 1:
                    core_in = observations[key].squeeze(0)
                    # print("img device nb", core_in.device)
                    core_out = self.img_cnn(core_in)
                    lstm_out = self.img_lstm(core_out)[-1][0]
                    encoded_tensor_list.append(lstm_out[-1].unsqueeze(0))
                else:
                    core_in = observations[key].to(model_device)
                    # print("img device batch", core_in.device)
                    batch_size = core_in.shape[0]
                    mini_batch_size = 32
                    num_mini_batches = batch_size // mini_batch_size
                    core_in = core_in.view(
                        num_mini_batches, mini_batch_size, -1, 3, 64, 64
                    )

                    core_out_n = []
                    for i in range(num_mini_batches):
                        cnn_out = self.img_cnn(core_in[i].view(-1, 3, 64, 64))
                        core_out_n.append(
                            cnn_out.view(
                                mini_batch_size, self.img_context, 256
                            ).swapaxes(0, 1)
                        )

                    core_out = th.cat(core_out_n, dim=1)

                    lstm_out = self.img_lstm(core_out)[-1][0]
                    encoded_tensor_list.append(lstm_out[-1])

        return th.cat(
            encoded_tensor_list, dim=1
        )  # encoded tensor is the batch dimension


def create_model(algo: str, my_env, gpus, policy_kwargs: dict, tensor_log_dir: str):
    if algo.lower() == "ppo":
        return PPO(
            PPOMultiPolicy,
            my_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tensor_log_dir,
            **env_config.training.ppo_config
        )
    elif algo.lower() == "sac":
        return SAC(
            SACMultiPolicy,
            my_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=f"cuda:{gpus[0]}",
            tensorboard_log=tensor_log_dir,
            **env_config.training.sac_config
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")