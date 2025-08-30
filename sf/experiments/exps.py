from utils import register_custom_env_envs
from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

register_custom_env_envs()

_params = ParamGrid(
    [
        ("seed", [1111]),
        ("use_rnn", [True]),
        ("batch_size", [2048]),
        ("glaucoma_level", [50, 100, 150, 200, 250, 300]),
        ("reward_type", ["rnd", "extrinsic"]),
        ("env", ["health_gathering_glaucoma"]),
    ]
)

_experiments = [
    Experiment(
        "glaucoma",
        "uv run sf/train.py --algo=APPO --train_for_env_steps=30000000 --num_workers=5 --num_envs_per_worker=4 --num_policies=1 --device=gpu",
        _params.generate_params(randomize=False),
    ),
]

RUN_DESCRIPTION = RunDescription("doom_health_gathering", experiments=_experiments)
