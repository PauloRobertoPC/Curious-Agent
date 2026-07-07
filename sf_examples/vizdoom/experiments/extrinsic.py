from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

viz_params = ParamGrid(
    [
        ("seed", [42]),
        ("decay_speed", [0, 10, 20, 30, 40, 50, 100]),
    ]
)

train_params = ParamGrid(
    [
        ("seed", [42, 43, 44, 45, 46]),
        ("decay_speed", [0, 10, 20, 30, 40, 50, 100]),
    ]
)

experiments = [
    Experiment(
        "visualize",
        "python -m sf_examples.vizdoom.train_custom_vizdoom_env "
        "--env custom_health_gathering "
        "--experiment rnd_reward_only_v "
        "--train_for_env_steps 10000000 "
        "--num_workers 4 "
        "--num_envs_per_worker 2 "
        "--steps_until_decay 0 "
        "--with_curiosity false "
        "--intrinsic_reward_coeff 0.0 "
        "--rnd_ext_coef 1.0 "
        "--env_frameskip 1 "
        "--save_every_env_steps 100000 "
        "--tot_envs_to_evaluate 4",
        viz_params.generate_params(randomize=False),
    ),
    Experiment(
        "train",
        "python -m sf_examples.vizdoom.train_custom_vizdoom_env "
        "--env custom_health_gathering "
        "--experiment rnd_reward_only_t "
        "--train_for_env_steps 10000000 "
        "--num_workers 4 "
        "--num_envs_per_worker 2 "
        "--steps_until_decay 0 "
        "--with_curiosity true "
        "--curiosity_module_type rnd "
        "--intrinsic_reward_coeff 0.0 "
        "--rnd_ext_coef 1.0"
        "--save_every_env_steps -1 "
        "--tot_envs_to_evaluate 4",
        train_params.generate_params(randomize=False),
    ),
]

RUN_DESCRIPTION = RunDescription(
    "extrinsic_reward_eval",
    experiments=experiments,
)
