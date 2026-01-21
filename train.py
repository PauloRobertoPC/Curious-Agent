from typing import List, Callable
from dataclasses import dataclass

import torch

from gymnasium.wrappers import TransformReward
from gymnasium.vector import VectorEnv

from rllte.agent import PPO
from rllte.env import health_gathering
from rllte.common.prototype import BaseReward
from rllte.xplore.reward import RND, NGU

from wrappers import image_wrapper, glaucoma_wrapper

WrapperFactory = Callable[[object], object]
IRSFactory = Callable[[VectorEnv], BaseReward]


def transform_reward_wrapper(f):
    return lambda env: TransformReward(env, f)

@dataclass
class experiment:
    device: str
    wrappers: List[WrapperFactory]
    tag: str
    irs: IRSFactory | None

def train(e:experiment):
    # creating envs
    env = health_gathering(num_envs=8, device=e.device, asynchronous=False, seed=1, wrappers=e.wrappers)
    eval_env = health_gathering(num_envs=8, device=e.device, asynchronous=False, seed=9, wrappers=e.wrappers)
    # create agent
    agent = PPO(env=env, 
                eval_env=eval_env, 
                device=device,
                tag=e.tag,
                )
    # create intrinsic reward
    if e.irs != None:
        irs = e.irs(env)
        agent.set(reward=irs)
    # start training
    agent.train(num_train_steps=8_000_000, save_interval=1)

if __name__ == "__main__":
    # which device to use while training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wrappers = [
        image_wrapper((84, 84)),
        # glaucoma_wrapper(0, ),
        transform_reward_wrapper(lambda r: 0)
    ]

    experiments: List[experiment] = []
    experiments.append(experiment(tag="100g_rnd", irs=lambda env: RND(env, lr=1e-5, device=device), wrappers=wrappers, device=device))
        
    for e in experiments:
        train(e)