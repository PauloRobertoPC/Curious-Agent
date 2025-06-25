from utils import register_custom_env_envs
from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

register_custom_env_envs()

_params = ParamGrid(
    [
        ("seed", [1111, 2222]),
        ("train_for_env_steps", [20_000_000]),
        ("algo", ["APPO"]),
        ("use_rnn", [True]),
        ("batch_size", [2048]),
        ("glaucoma_level", [50, 100, 150, 200, 250, 300]),
        ("env", ["health_gathering_glaucoma"]),
    ]
)

_experiments = [
    Experiment(
        "glaucoma",
        "uv run sf/train.py --num_workers=5 --num_envs_per_worker=4 --num_policies=1 --device=gpu --obs_scale=255.0",
        _params.generate_params(randomize=False),
    ),
]

RUN_DESCRIPTION = RunDescription("doom_health_gathering", experiments=_experiments)
