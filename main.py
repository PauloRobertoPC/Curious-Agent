import envpool
from rllte.env.utils import EnvPoolSync2Gymnasium, Gymnasium2Torch
from wrappers.glaucoma import GlaucomaWrapper
from wrappers.image_transformation import ImageTransformationWrapper

env_kwargs = dict(
    task_id="VizdoomCustom-v1",
    cfg_path="scenarios/health_gathering.cfg",
    wad_path="scenarios/health_gathering.wad",
    env_type="gymnasium",
    num_envs=1,
    batch_size=1,
    seed=1,
    episodic_life=True,
    use_combined_action=True,
    stack_num=1
)

envs = EnvPoolSync2Gymnasium(env_kwargs)
print(type(envs))
# envs = ImageTransformationWrapper(envs, (84, 84))
# envs = GlaucomaWrapper(envs, 0, 5, -100)
# envs = Gymnasium2Torch(envs, device="cuda", envpool=True)
print(envs.action_space)
print(envs.observation_space)