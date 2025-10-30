from utils import register_custom_env_envs
from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

register_custom_env_envs()

_params = ParamGrid(
    [
        ("seed", [2222]),
        ("batched_sampling", [True, False]),
    ]
)

_experiments = [
    Experiment(
        "glaucoma",
        "uv run sf/train.py --reward_type=extrinsic --glaucoma_level=150 --algo=APPO --train_for_env_steps=10000000 --num_workers=8 --num_envs_per_worker=4 --device=gpu --env=health_gathering_glaucoma",
        _params.generate_params(randomize=False),
    ),
]

RUN_DESCRIPTION = RunDescription("doom_health_gathering", experiments=_experiments)
