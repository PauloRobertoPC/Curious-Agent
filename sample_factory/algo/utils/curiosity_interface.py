"""
Defines the abstract interface for Curiosity Modules.

This file establishes the contract that any intrinsic reward algorithm
(such as RND, ICM, etc.) must follow in order to integrate with the
Sample Factory Learner in a plug-and-play manner.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import torch
from torch import Tensor


class CuriosityModule(ABC):
    """Base interface for curiosity modules."""

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_rewards(self, obs_dict: Dict[str, Tensor], dones: Tensor) -> Tensor:
        """Returns a tensor of intrinsic rewards with shape (batch_size,). Terminal transitions indicated by `dones` should be ignored."""
        pass

    @abstractmethod
    def update(self, obs_dict: Dict[str, Tensor], dones: Tensor) -> Tensor:
        """Trains the curiosity model and returns the training loss (scalar)."""
        pass

    @abstractmethod
    def get_checkpoint_dict(self) -> Dict[str, Any]:
        """Returns the state required to save the module in a checkpoint."""
        pass

    @abstractmethod
    def load_checkpoint_dict(self, checkpoint_dict: Dict[str, Any]) -> None:
        """Restores the module state from a checkpoint."""
        pass
