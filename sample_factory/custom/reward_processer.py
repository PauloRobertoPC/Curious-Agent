from abc import ABC, abstractmethod

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.typing import Config

from torch import nn

class RewardProcesser(ABC, nn.Module, Configurable):

    def __init__(self, cfg: Config, env_info: EnvInfo):
        nn.Module.__init__(self)
        Configurable.__init__(self, cfg)
        self.env_info = env_info
        self.train_times = 0

    @abstractmethod
    def calculate_reward(self):
        pass

    @abstractmethod
    def train(self, batch:TensorDict):
        pass

class Extrinsic(RewardProcesser):

    def __init__(self, cfg: Config, env_info: EnvInfo):
        super().__init__(cfg, env_info)

    def calculate_reward(self):
        pass

    def train(self, batch:TensorDict):
        self.train_times += 1

class RND(RewardProcesser):

    def __init__(self, cfg: Config, env_info: EnvInfo):
        super().__init__(cfg, env_info)

    def calculate_reward(self):
        pass

    def train(self, batch:TensorDict):
        pass