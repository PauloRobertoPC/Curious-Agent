# Curious Agent

## Sample Factory

### Training an agent

```bash 
uv run python -m sf_examples.vizdoom.train_custom_vizdoom_env \
  --env custom_health_gathering \
  --experiment smoke_test \
  --train_for_env_steps 10000 \
  --num_workers 4 --num_envs_per_worker 2 \
  --steps_until_decay 0 --decay_speed 20 \
  --with_curiosity false --curiosity_module_type rnd \
  --calculate_agent_trajectory true \
  --tot_envs_to_evaluate 4 --env_frameskip 1
```

### Running experiments(Grid Search)

If you want to execute more experiments, then add an experiment file into sf_examples/vizdoom/experiments

```bash 
uv run python -m sample_factory.launcher.run \
   --run=sf_examples.vizdoom.experiments.extrinsic \
   --backend=processes --max_parallel=1  --pause_between=1 \
   --experiments_per_gpu=1 --num_gpus=1
```

###  Agent playing
```bash 
uv run python -m sf_examples.vizdoom.enjoy_custom_vizdoom_env \
  --env custom_health_gathering \
  --experiment rnd_reward_eval/rnd_reward_/00_rnd_reward_i.r.coe_1.0_d.spe_40
```

###  Play yourself
```bash 
uv run python -m  sf_examples.vizdoom.play_human \
  --env custom_health_gathering \
  --scenario_cfg health_gathering.cfg \
  --steps_until_decay 0 --decay_speed 10 \
  --game_layout 0
```

### Generate data for experiment
```bash
uv run python -m  sf_examples.vizdoom.generate_data_from_experiment \
  --env custom_health_gathering \
  --experiment rnd_reward_eval/visualize_/00_visualize_see_42_d.spe_0 \
  --tot_envs_to_evaluate 4
```
```

### To see metrics in tensorboard run
```bash 
uv run -m tensorboard.main --logdir=./train_dir
```

ADD A GREEN POINT IN THE REWARD GRAPH WHEN MEDKIT WAS EATEN AND PURPLE ONE WHEN A POINSON WAS EATEN
READ MORE ABOUT EXPERIMENTS CLI ARGUMENTS
EXPLAIN WHAT KIND OF CHANGES WERE MADE IF COMPARED TO THE DEFAULT SF
EXPLAIN DEFAULT ARGUMENTS
    - doom_override_defaults functions changes the default arguments
        use_rnn=False, # added by me
        ppo_clip_value=0.2,  # value used in all experiments in the paper
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        exploration_loss="symmetric_kl",
        exploration_loss_coeff=0.001,
        normalize_returns=True,
        normalize_input=True,
        env_frameskip=4,
        eval_env_frameskip=1,  # this is for smoother rendering during evaluation
        fps=35,  # for evaluation only
        heartbeat_reporting_interval=600,

uv run python -m  sf_examples.vizdoom.generate_data_from_experiment \
  --env custom_health_gathering \
  --experiment rnd_reward_eval/visualize_/00_visualize_see_42_d.spe_0 \
  --tot_envs_to_evaluate 4
