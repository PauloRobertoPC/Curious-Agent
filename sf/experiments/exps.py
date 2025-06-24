from utils import register_custom_env_envs
from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

register_custom_env_envs()

_params = ParamGrid(
    [
        ("seed", [1111, 2222]),
        ("env", ["health_gathering_glaucoma"]),
    ]
)

_experiments = [
    Experiment(
        "glaucoma_50",
        "uv run sf/train.py --train_for_env_steps=40_000_000 --algo=APPO --use_rnn=True --num_workers=4 --num_envs_per_worker=4 --num_policies=1 --batch_size=2048 --device=gpu --obs_scale=255.0 --glaucoma_level=50",
        _params.generate_params(randomize=False),
    ),
]

RUN_DESCRIPTION = RunDescription("doom_health_gathering", experiments=_experiments)
