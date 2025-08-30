from abc import ABC, abstractmethod

from rlexplore.rnd.rnd import RND
from sample_factory.algo.utils.env_info import obtain_env_info_in_a_separate_process

class BatchProcesser(ABC):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        
    @abstractmethod
    def process(self, batch):
        pass
    
class ExtrinsicRewardProcesser(BatchProcesser):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        
    def process(self, batch):
        return batch

class ZeroRewardProcesser(BatchProcesser):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        
    def process(self, batch):
        batch["rewards"].zero_()
        return batch


def rnd_instance(cfg):
    env_info = obtain_env_info_in_a_separate_process(cfg)
    return RND(env_info.obs_space["obs"].shape, env_info.action_space, "cuda", 1024, 0.001, cfg.batch_size//cfg.rollout, 1, 0.001)

exploration_maker = {}
exploration_maker["rnd"] = rnd_instance

class ExplorationRewardProcesser(BatchProcesser):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.time_step = 0
        self.exploration_calculator = exploration_maker[cfg.reward_type](cfg)
    
    def process(self, batch):
        intrinsic_reward = self.exploration_calculator.compute_irs({'observations': batch["obs"]["obs"][:, :-1]}, self.time_step)
        batch["rewards"] = intrinsic_reward
        self.time_step += self.cfg.rollout
        return batch

reward_maker = {}
reward_maker["extrinsic"] = ExtrinsicRewardProcesser
reward_maker["zero"] = ZeroRewardProcesser
reward_maker["exploration"] = ExplorationRewardProcesser

