from .reward_processer import RewardProcesser, Extrinsic, RND

CLASS_REGISTRY = {
    "extrinsic": Extrinsic,
    "rnd": RND,
}

def create_instance(name: str, *args, **kwargs):
    if name not in CLASS_REGISTRY:
        raise ValueError(f"Reward not known: {name}")
    return CLASS_REGISTRY[name](*args, **kwargs)
