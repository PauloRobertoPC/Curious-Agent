#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：rnd.py
@Author ：YUAN Mingqi
@Date ：2022/9/20 21:46 
'''

from rlexplore.networks.random_encoder import CnnEncoder, MlpEncoder
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

class RND(object):
    def __init__(self,
                 obs_shape,
                 action_shape,
                 device,
                 latent_dim,
                 lr,
                 batch_size,
                 beta,
                 kappa
                 ):
        """
        Exploration by Random Network Distillation (RND)
        Paper: https://arxiv.org/pdf/1810.12894.pdf

        :param obs_shape: The data shape of observations.
        :param action_shape: The data shape of actions.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param latent_dim: The dimension of encoding vectors of the observations.
        :param lr: The learning rate of predictor network.
        :param batch_size: The batch size to train the predictor network.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.kappa = kappa

        if len(self.obs_shape) == 3:
            self.predictor = CnnEncoder(obs_shape, latent_dim)
            self.target = CnnEncoder(obs_shape, latent_dim)
        else:
            self.predictor = MlpEncoder(obs_shape, latent_dim)
            self.target = MlpEncoder(obs_shape, latent_dim)

        self.predictor.to(self.device)
        self.target.to(self.device)

        self.loss = nn.MSELoss()
        self.opt = optim.Adam(self.predictor.parameters(), self.lr)

        # freeze the network parameters
        for p in self.target.parameters():
            p.requires_grad = False
            
    def compute_irs(self, rollouts, time_steps):
        aux = self.test(rollouts, time_steps)
        
        obs_tensor = rollouts['observations'].to(self.device)  # shape: (n_steps, n_envs, *obs_shape)
        n_steps, n_envs = obs_tensor.shape[:2]
        obs_flat = obs_tensor.reshape(n_steps * n_envs, *obs_tensor.shape[2:])  # flatten time & env
        self.update(obs_flat)
        
        return aux
        
    @torch.no_grad()
    def test(self, rollouts, time_steps):
        """
        Compute the intrinsic rewards using the collected observations (PyTorch only).
        :param rollouts: The collected experiences (with 'observations' as a torch.Tensor).
        :param time_steps: The current time step (scalar or tensor).
        :return: Intrinsic rewards as torch.Tensor of shape (n_steps, n_envs)
        """
        self.predictor.eval()

        # Weighting coefficient at this timestep
        beta_t = self.beta * (1.0 - self.kappa) ** time_steps

        obs_tensor = rollouts['observations'].to(self.device)  # shape: (n_steps, n_envs, *obs_shape)
        n_steps, n_envs = obs_tensor.shape[:2]

        intrinsic_rewards = torch.zeros(n_steps, n_envs, 1, device=self.device)

        with torch.no_grad():
            for idx in range(n_envs):
                src_feats = self.predictor(obs_tensor[:, idx])  # (n_steps, latent_dim)
                tgt_feats = self.target(obs_tensor[:, idx])     # (n_steps, latent_dim)

                dist = F.mse_loss(src_feats, tgt_feats, reduction='none').mean(dim=1)  # (n_steps,)
                dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-11)         # normalize
                intrinsic_rewards[:-1, idx, 0] = dist[1:]  # shift by 1

        return beta_t * intrinsic_rewards.squeeze(-1)  # shape: (n_steps, n_envs)


    @torch.enable_grad()
    def update(self, obs):
        dataset = TensorDataset(obs)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=True)
        
        self.predictor.train()
        for idx, batch_data in enumerate(loader):
            batch_obs = batch_data[0]
            src_feats = self.predictor(batch_obs)
            tgt_feats = self.target(batch_obs)

            loss = self.loss(src_feats, tgt_feats)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
