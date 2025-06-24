import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

import gymnasium as gym

class CnnEncoder(nn.Module):
    def __init__(self, obs_shape, latent_dim):
        super(CnnEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, (8, 8), stride=(4, 4)), nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=(2, 2)), nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3), stride=(1, 1)), nn.ReLU(), nn.Flatten(),
            # nn.Linear(32 * 7 * 7, latent_dim), nn.LayerNorm(latent_dim))
            # TODO: FIX IT TO LET IT THE "8192 below automatic"
            nn.Linear(8192, latent_dim), nn.LayerNorm(latent_dim))

    def forward(self, ob):
        x = self.main(ob)

        return x

class MlpEncoder(nn.Module):
    def __init__(self, obs_shape, latent_dim):
        super(MlpEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(obs_shape[0], 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.LayerNorm(latent_dim))

    def forward(self, ob):
        x = self.main(ob)

        return x


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

        self.predictor = CnnEncoder(obs_shape, latent_dim)
        self.target = CnnEncoder(obs_shape, latent_dim)

        self.predictor.to(self.device)
        self.target.to(self.device)

        self.opt = optim.Adam(lr=self.lr, params=self.predictor.parameters())

        self.count = 0
        
        # freeze the network parameters
        for p in self.target.parameters():
            p.requires_grad = False

    def compute_irs(self, rollouts, time_steps=1):
        """
        Compute the intrinsic rewards using the collected observations.
        :param rollouts: The collected experiences.
        :param time_steps: The current time steps.
        :return: The intrinsic rewards
        """

        # print("****************************************************************************************************")
        # print(rollouts['observations'].shape)
        # print("****************************************************************************************************")
        # compute the weighting coefficient of timestep t
        self.count += 1
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        n_steps = rollouts['observations'].shape[0]
        n_envs = rollouts['observations'].shape[1]

        # observations shape ((n_steps, n_envs) + obs_shape)
        obs_tensor = torch.from_numpy(rollouts['observations'])
        obs_tensor = obs_tensor.to(self.device)

        dist = 0
        with torch.no_grad():
            src_feats = self.predictor(obs_tensor[:, 0])
            tgt_feats = self.target(obs_tensor[:, 0])
            dist = F.mse_loss(src_feats, tgt_feats, reduction='none').mean(dim=1)

        # update the predictor network
        self.update(torch.clone(obs_tensor).reshape(n_steps*n_envs, *obs_tensor.size()[2:]))

        # return beta_t * dist
        return  dist.cpu().item()

    def update(self, obs):
        dataset = TensorDataset(obs)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=False)
        
        self.predictor.train()
        for idx, batch_data in enumerate(loader):
            batch_obs = batch_data[0]
            src_feats = self.predictor(batch_obs)
            tgt_feats = self.target(batch_obs)

            loss = F.mse_loss(src_feats, tgt_feats)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

class IntrinsicRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.rnd = RND(
                        env.observation_space.shape, 
                        env.action_space.shape,
                        "cuda" if torch.cuda.is_available() else "cpu",
                        # "cpu",
                        512,
                        0.001,
                        1,
                        1,
                        1
                    )

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # return self.env.step(action)
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        obs2 = obs.astype(np.float32)/255
        obs2 = obs2[np.newaxis, np.newaxis, :]
        intrinsic_reward = self.rnd.compute_irs({"observations": obs2})
        return obs, intrinsic_reward, terminated, truncated, info
