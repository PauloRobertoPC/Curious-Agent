from typing import Dict, Any, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.algo.utils.curiosity_interface import CuriosityModule
from sample_factory.algo.utils.running_mean_std import RunningMeanStd, RunningMeanStdInPlace


class RNDEncoder(nn.Module):
    """
    CNN encoder based on the 'Nature CNN' architecture (Mnih et al., 2015).
    Commonly used in RND implementations for Atari and VizDoom.
    """

    def __init__(self, obs_space: ObsSpace, latent_dim: int = 512):
        super().__init__()

        # Get observation dimensions (C, H, W)
        obs_key = "obs"
        shape = obs_space[obs_key].shape
        c = shape[0]

        # Three-layer Nature CNN architecture:
        # 8x8 kernel (stride 4) -> 4x4 kernel (stride 2) -> 3x3 kernel (stride 1)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Dynamically compute the CNN output size.
        # This allows the encoder to work with different input resolutions
        # (e.g., 84x84 or 240x320) without modification.
        with torch.no_grad():
            dummy_obs = torch.zeros(1, *shape)
            output_dim = self.feature_extractor(dummy_obs).shape[1]

        # Final projection layers
        self.head = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),  # Additional normalization for stability
        )

    def forward(self, obs_tensor: Tensor) -> Tensor:
        x = self.feature_extractor(obs_tensor)
        return self.head(x)


class RNDModule(CuriosityModule, nn.Module):
    """
    Complete Random Network Distillation (RND) module.
    """

    def __init__(self, cfg: Config, obs_space: ObsSpace, device: torch.device):
        CuriosityModule.__init__(self)
        nn.Module.__init__(self)

        self.cfg = cfg
        self.device = device

        # 1. Observation normalizer (input normalization)
        obs_key = "obs"
        self.obs_rms = RunningMeanStd(input_shape=obs_space[obs_key].shape).to(device)

        # 2. Neural networks (Predictor and Target)
        self.predictor = RNDEncoder(obs_space).to(device)
        self.target = RNDEncoder(obs_space).to(device)

        # Freeze the target network (fixed random network)
        for p in self.target.parameters():
            p.requires_grad = False

        # 3. Optimizer
        self.rnd_optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=getattr(cfg, "rnd_lr", 1e-4),
        )

        # 4. Intrinsic reward normalizer (output normalization)
        # norm_only=True ensures that only the running standard deviation is used,
        # preserving the positivity of the intrinsic reward.
        self.intrinsic_reward_rms = RunningMeanStdInPlace(
            input_shape=(1,),
            norm_only=True,
        ).to(device)

    def _normalize_obs(self, obs: Tensor) -> Tensor:
        """Normalizes and clips the input observation."""

        # Ensure the observation is on the correct device
        if obs.device != self.obs_rms.running_mean.device:
            obs = obs.to(self.obs_rms.running_mean.device)

        normalized = self.obs_rms(obs)
        normalized = torch.clamp(normalized, -5, 5)
        return normalized

    def _flatten_5d(self, obs: Tensor, dones: Optional[Tensor] = None):
        """
        Flattens observations from [B, T+1, ...] to [B*(T+1), ...]
        and returns the original shape.

        Since `dones` has shape [B, T], it is padded with False values
        before being flattened.
        """
        orig_shape = None

        if obs.ndim == 5:
            orig_shape = obs.shape[:2]  # [B, T+1]
            obs = obs.flatten(0, 1)     # [B*(T+1), C, H, W]

            if dones is not None:
                B, T = dones.shape[0], dones.shape[1]
                T1 = orig_shape[1]  # T+1

                if T < T1:
                    # Pad because obs[T+1] corresponds to the bootstrap next observation
                    pad = torch.zeros(B, T1 - T, dtype=dones.dtype, device=dones.device)
                    dones = torch.cat([dones, pad], dim=1)

                dones = dones.flatten(0, 1)

        return obs, dones, orig_shape

    def _build_valid_mask(self, dones: Optional[Tensor], n: int) -> Optional[Tensor]:
        """Creates a boolean mask for valid (non-terminal) observations."""

        if dones is None:
            return None

        mask = ~dones.bool()

        if mask.ndim > 1:
            mask = mask.squeeze(-1)

        return mask[:n]

    def eval(self):
        super().eval()

        self.predictor.eval()
        self.target.eval()
        self.obs_rms.eval()
        self.intrinsic_reward_rms.eval()

        return self

    def calculate_rewards(
        self,
        obs_dict: Dict[str, Tensor],
        dones: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Computes intrinsic rewards.

        Terminal observations (e.g., black terminal frames) are ignored.
        A 5D input tensor [Batch, Time, C, H, W] is automatically flattened.
        """

        obs = obs_dict["obs"]
        obs, dones_flat, orig_shape = self._flatten_5d(obs, dones)

        n = obs.shape[0]
        valid_mask = self._build_valid_mask(dones_flat, n)

        with torch.no_grad():
            # Filter valid observations to avoid contaminating obs_rms
            # with terminal black frames.
            if valid_mask is not None and valid_mask.sum() > 0 and valid_mask.sum() < n:
                obs_valid = obs[valid_mask]
            else:
                obs_valid = obs

            normalized_obs = self._normalize_obs(obs_valid)

            target_feature = self.target(normalized_obs)
            predictor_feature = self.predictor(normalized_obs)

            valid_reward = F.mse_loss(
                predictor_feature,
                target_feature,
                reduction="none",
            ).mean(dim=-1)

            # Normalize intrinsic rewards in-place by dividing
            # by the running standard deviation.
            reward_for_norm = valid_reward.unsqueeze(-1)
            self.intrinsic_reward_rms(reward_for_norm)
            valid_reward = reward_for_norm.squeeze(-1)

            # Reconstruct the complete reward tensor.
            # Terminal observations receive zero intrinsic reward.
            if valid_mask is not None and valid_mask.sum() < n:
                full_reward = torch.zeros(n, device=valid_reward.device)
                full_reward[valid_mask] = valid_reward
            else:
                full_reward = valid_reward

        if orig_shape is not None:
            full_reward = full_reward.view(orig_shape)

        return full_reward

    def update(self, obs_dict: Dict[str, Tensor], dones: Tensor) -> Tensor:
        """
        Updates the predictor network using only valid (non-terminal)
        observations.

        The observation normalizer is not updated here since it has already
        been updated during reward computation.
        """

        obs = obs_dict["obs"]
        obs, dones_flat, _ = self._flatten_5d(obs, dones)

        n = obs.shape[0]
        valid_mask = self._build_valid_mask(dones_flat, n)

        # Keep only valid observations
        if valid_mask is not None and valid_mask.sum() > 0 and valid_mask.sum() < n:
            obs = obs[valid_mask]

        # Temporarily switch obs_rms to evaluation mode so that
        # its running statistics remain unchanged.
        was_training = self.obs_rms.training
        self.obs_rms.eval()

        normalized_obs = self._normalize_obs(obs)

        if was_training:
            self.obs_rms.train()

        predictor_feature = self.predictor(normalized_obs)

        with torch.no_grad():
            target_feature = self.target(normalized_obs)

        loss = F.mse_loss(predictor_feature, target_feature)

        self.rnd_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)
        self.rnd_optimizer.step()

        return loss

    def get_checkpoint_dict(self) -> Dict[str, Any]:
        return {
            "predictor": self.predictor.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.rnd_optimizer.state_dict(),
            "obs_rms": self.obs_rms.state_dict(),
            "reward_rms": self.intrinsic_reward_rms.state_dict(),
        }

    def load_checkpoint_dict(self, checkpoint_dict: Dict[str, Any]) -> None:
        if "predictor" in checkpoint_dict:
            self.predictor.load_state_dict(checkpoint_dict["predictor"])
            self.target.load_state_dict(checkpoint_dict["target"])
            self.rnd_optimizer.load_state_dict(checkpoint_dict["optimizer"])
            self.obs_rms.load_state_dict(checkpoint_dict["obs_rms"])
            self.intrinsic_reward_rms.load_state_dict(checkpoint_dict["reward_rms"])
