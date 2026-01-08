import os
from pathlib import Path
import sys
from typing import Sequence, Dict, Union
from abc import ABC, abstractmethod

import torch

from stable_baselines3.common.callbacks import BaseCallback

CHECKPOINT_FREQUENCY = 100000

class EnvSetup(ABC):
    def __init__(self, scenario:str, tot_actions:int, obs_space_shape:Sequence[int]):
        self.train_and_logging_callback = None 
        self.scenario = f"./scenarios/{scenario}"
        self.tot_actions = tot_actions
        self.obs_space_shape: Sequence[int] = obs_space_shape
        self.new_number = -1

    @abstractmethod
    def set_train_and_logging_callback(self):
        raise NotImplementedError()

    @abstractmethod
    def get_info(self, state):
        raise NotImplementedError()

    @abstractmethod
    def zero_info(self):
        raise NotImplementedError()

    @abstractmethod
    def done_info(self, info):
        raise NotImplementedError()

    @abstractmethod
    def make_env(self):
        raise NotImplementedError()
    
    @abstractmethod
    def _get_train_dir(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def _get_log_dir(self) -> str:
        raise NotImplementedError()

    def get_train_dir(self) -> str:
        if self.new_number == -1:
            return self._get_train_dir()
        return f"{self._get_train_dir()}_{self.new_number}"

    def get_log_dir(self) -> str:
        if self.new_number == -1:
            return self._get_log_dir()
        return f"{self._get_log_dir()}_{self.new_number}"

    def create_directory(self):
        i = 1;
        while True:
            train_dir = f"{self._get_train_dir()}_{i}"
            log_dir = f"{self._get_log_dir()}_{i}"

            dir_path_train = Path(train_dir)
            dir_path_log = Path(log_dir)

            directory_train_exists = dir_path_train.exists() and dir_path_train.is_dir()
            directory_log_exists = dir_path_log.exists() and dir_path_log.is_dir()
            if not directory_train_exists and not directory_log_exists:
                self.new_number = i
                os.makedirs(train_dir, exist_ok=True)
                os.makedirs(log_dir, exist_ok=True)
                print(f"Directories {train_dir} and {log_dir} were created")
                break
            i += 1

def play_episode(env, model):
    total_reward = 0
    finished = False
    info = {}
    obs, _ = env.reset()
    ep_len = 0
    while not finished:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        ep_len += 1
        finished = done or truncated
    return 4*ep_len, info