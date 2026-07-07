from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("intrinsic_reward_coeff", [1.0]),
        ("decay_speed", [40, 50, 60]),
    ]
)

_experiments = [
    Experiment(
        "rnd_reward",
        "python -m sf_examples.vizdoom.train_custom_vizdoom_env "
        "--env custom_health_gathering "
        "--experiment rnd_reward_only "
        "--train_for_env_steps 10000000 "
        "--num_workers 4 "
        "--num_envs_per_worker 2 "
        "--steps_until_decay 0 "
        "--with_curiosity true "
        "--curiosity_module_type rnd "
        "--rnd_ext_coef 0.0",
        _params.generate_params(randomize=False),
    )
]

RUN_DESCRIPTION = RunDescription(
    "rnd_reward_eval",
    experiments=_experiments,
)
