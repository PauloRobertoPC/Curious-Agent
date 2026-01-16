import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.wrappers import NormalizeReward, TransformReward

from rllte.env.utils import Gymnasium2Torch
from rllte.env.utils import EnvPoolAsync2Gymnasium, EnvPoolSync2Gymnasium, Gymnasium2Torch

from wrappers.glaucoma import GlaucomaWrapper
from wrappers.image_transformation import ImageTransformationWrapper

def make_envpool_vizdoom_env(
    env_id: str = "MyWayHome-v1", num_envs: int = 8, device: str = "cpu", seed: int = 1, asynchronous: bool = True
) -> Gymnasium2Torch:
    env_kwargs = dict(
        task_id=env_id,
        env_type="gymnasium",
        num_envs=num_envs,
        batch_size=num_envs,
        seed=seed,
        episodic_life=True,
        use_combined_action=True,
        stack_num=1
    )

    if asynchronous:
        envs = EnvPoolAsync2Gymnasium(env_kwargs)
    else:
        envs = EnvPoolSync2Gymnasium(env_kwargs)

    envs = RecordEpisodeStatistics(envs)
    return Gymnasium2Torch(envs, device, envpool=True)

def make_envpool_vizdoom_env_custom(
    cfg_path:str, wad_path:str, num_envs: int, device: str, seed: int, asynchronous: bool
) -> Gymnasium2Torch:
    env_kwargs = dict(
        task_id="VizdoomCustom-v1",
        cfg_path=cfg_path,
        wad_path=wad_path,
        env_type="gymnasium",
        num_envs=num_envs,
        batch_size=num_envs,
        seed=seed,
        episodic_life=True,
        use_combined_action=True,
        stack_num=1
    )

    if asynchronous:
        envs = EnvPoolAsync2Gymnasium(env_kwargs)
    else:
        envs = EnvPoolSync2Gymnasium(env_kwargs)

    # Custom Wrappers
    envs = ImageTransformationWrapper(envs, (84, 84))
    envs = GlaucomaWrapper(envs, 0, 5)

    envs = RecordEpisodeStatistics(envs)
    return Gymnasium2Torch(envs, device, envpool=True)


def health_gathering(num_envs: int = 8, device: str = "cpu", seed: int = 1, asynchronous: bool = True):
    return make_envpool_vizdoom_env_custom(
        "scenarios/health_gathering.cfg",
        "scenarios/health_gathering.wad",
        num_envs=num_envs,
        device=device,
        seed=seed,
        asynchronous=asynchronous
    )
