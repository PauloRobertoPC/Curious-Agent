import torch
from torch import nn
import torch.nn.functional as F

import gymnasium as gym
import numpy as np


class RNDModel(nn.Module):
    def __init__(self):
        super().__init__()

        feature_output = 8192
        latent_output = 512

        self.predictor = nn.Sequential(
            nn.Conv2d(1, 32, 8, 4),
            nn.ELU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(feature_output, latent_output),
            nn.ELU()
        )

        self.target = nn.Sequential(
            nn.Conv2d(1, 32, 8, 4),
            nn.ELU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ELU(),
            nn.Conv2d(64, 128, 3, 2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(feature_output, latent_output),
            nn.ELU()
        )

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, next_obs: torch.Tensor):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)
        return F.mse_loss(predict_feature, target_feature, reduction='none').mean(dim=1)


class RNDWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, batch_size: int = 32, learning_rate: float = 1e-4, device: str = 'cpu'):
        super().__init__(env)
        self.device = device
        self.batch_size = batch_size

        self.rnd_model = RNDModel().to(device)
        self.optimizer = torch.optim.Adam(self.rnd_model.predictor.parameters(), lr=learning_rate)

        self.obs_buffer = []
        self.steps_done = 0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device).float()

        with torch.no_grad():
            intrinsic_reward = self.rnd_model(obs_tensor).item()

        new_reward = 1000*intrinsic_reward

        self.obs_buffer.append(obs_tensor)
        self.steps_done += 1

        if self.steps_done >= self.batch_size:
            batch = torch.cat(self.obs_buffer, dim=0)
            self.obs_buffer.clear()
            self.steps_done = 0

            loss = self.rnd_model(batch).mean()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return obs, new_reward, terminated, truncated, info
