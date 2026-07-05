from typing import Sequence
from abc import ABC, abstractmethod

class EnvSetup(ABC):
    def __init__(self, scenario:str, tot_actions:int, obs_space_shape:Sequence[int]):
        self.train_and_logging_callback = None 
        self.scenario = f"./scenarios/{scenario}"
        self.tot_actions = tot_actions
        self.obs_space_shape: Sequence[int] = obs_space_shape

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