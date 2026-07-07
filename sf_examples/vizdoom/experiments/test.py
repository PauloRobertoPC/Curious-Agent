from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

viz_params = ParamGrid(
    [
        ("seed", [42]),
        ("decay_speed", [20]),
    ]
)

_experiments = [
    Experiment(
        "no_poison",
        "python -m sf_examples.vizdoom.train_custom_vizdoom_env "
        "--env custom_health_gathering "
        "--experiment rnd_reward_only_v "
        "--train_for_env_steps 10000000 "
        "--num_workers 4 "
        "--num_envs_per_worker 2 "
        "--steps_until_decay 0 "
        "--with_curiosity true "
        "--curiosity_module_type rnd "
        "--intrinsic_reward_coeff 1.0 "
        "--rnd_ext_coef 0.0 "
        "--env_frameskip 1 "
        "--save_every_env_steps 100000 "
        "--tot_envs_to_evaluate 4",
        viz_params.generate_params(randomize=False),
    ),
    Experiment(
        "with_poison",
        "python -m sf_examples.vizdoom.train_custom_vizdoom_env "
        "--env custom_health_gathering "
        "--scenario_cfg health_gathering_poison.cfg "
        "--experiment rnd_reward_only_v "
        "--train_for_env_steps 100000000 "
        "--num_workers 4 "
        "--num_envs_per_worker 2 "
        "--steps_until_decay 0 "
        "--with_curiosity true "
        "--curiosity_module_type rnd "
        "--intrinsic_reward_coeff 1.0 "
        "--rnd_ext_coef 0.0 "
        "--env_frameskip 1 "
        "--save_every_env_steps 100000 "
        "--tot_envs_to_evaluate 4",
        viz_params.generate_params(randomize=False),
    ),
]

RUN_DESCRIPTION = RunDescription(
    "test",
    experiments=_experiments,
)
