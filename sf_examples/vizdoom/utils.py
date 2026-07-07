import os
from os.path import join, dirname, abspath

# Import Gymnasium to manually correct the action space
import gymnasium as gym
from gymnasium.spaces import Discrete

# Sample Factory Imports
from sample_factory.envs.env_utils import register_env
from sf_examples.vizdoom.doom.doom_gym import VizdoomEnv
from sf_examples.vizdoom.doom.wrappers.scenario_wrappers.gathering_reward_shaping import (
    DoomGatheringRewardShaping,
)

from sf_examples.vizdoom.doom.wrappers.image_transformation import ImageTransformationWrapper
from sf_examples.vizdoom.doom.wrappers.glaucoma import GlaucomaWrapper
from sf_examples.vizdoom.doom.wrappers.trajectory_visualization import TrajectoryVisualizationWrapper

def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("true", "1", "yes", "y")

def add_custom_args(parser):
    # game and wrappers arguments
    parser.add_argument("--scenario_cfg", type=str, default="health_gathering.cfg",
                        choices=["health_gathering.cfg", "health_gathering_poison.cfg"],
                        help="Which .cfg scenario vizdoom will load "
                             "'health_gathering.cfg' (default) "
                             "'health_gathering_supreme.cfg'")
    parser.add_argument("--game_layout", type=int, default=0,
                        choices=[0, 1, 2, 3, 4],
                        help="change in game medikits distribution "
                        "0 - random(default) "
                        "1 - square "
                        "2 - circle "
                        "3 - sin "
                        "4 - grid ",
                        )
    parser.add_argument(
        "--calculate_agent_trajectory",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--calculate_agent_trajectory_directory",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--calculate_agent_trajectory_step",
        type=int,
        default=1,
    )

    # glaucoma parameters
    parser.add_argument("--steps_until_decay", type=int, default=0,
                        help="Janela de tolerância (passos) antes da cegueira iniciar (Alostase)")
    parser.add_argument("--decay_speed", type=int, default=0,
                        help="Quantos pixels são apagados por passo após o fim da tolerância")

    # rnd parameters
    parser.add_argument("--with_curiosity", type=lambda x: str(x).lower() == 'true', default=False,
                        help="Ativa o módulo de curiosidade customizado")
    parser.add_argument("--curiosity_module_type", type=str, default='rnd',
                        help="Tipo do módulo (ex: rnd)")
    parser.add_argument("--intrinsic_reward_coeff", type=float, default=1.0,
                        help="Peso da recompensa intrínseca na otimização")
    parser.add_argument("--rnd_lr", type=float, default=1e-4,
                        help="Learning rate específico para a rede do RND")
    parser.add_argument("--rnd_ext_coef", type=float, default=0.0,
                        help="Coeficiente para a recompensa extrínseca (0.0 para ignorar recompensa do jogo)")

    # experiment parameters
    parser.add_argument(
        "--save_every_env_steps",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--tot_envs_to_evaluate",
        type=int,
        default=0,
    )



def make_custom_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    # scenario directory
    script_dir = dirname(abspath(__file__))
    scenarios_dir = join(script_dir, "doom", "scenarios")

    # choosing .cfg scenario
    scenario_cfg_name = getattr(cfg, 'scenario_cfg', 'health_gathering.cfg')
    config_path = join(scenarios_dir, scenario_cfg_name)

    # checking if the scenario exists
    if not os.path.exists(config_path):
        fallback_path = join(os.getcwd(), "sf_examples", "vizdoom", "doom", "scenarios", scenario_cfg_name)
        if os.path.exists(fallback_path):
            config_path = fallback_path
        else:
            raise FileNotFoundError(f"configuration file does not exists: {config_path}")

    game_layout = getattr(cfg, 'game_layout')

    calculate_agent_trajectory = getattr(cfg, "calculate_agent_trajectory")
    env = VizdoomEnv(
        action_space=Discrete(1+3),
        config_file=config_path,
        skip_frames=getattr(cfg, 'env_frameskip'), # 4 by default
        render_mode=render_mode,
        game_layout=game_layout,
        calculate_agent_trajectory=calculate_agent_trajectory
    )

    # # Match the stock health-gathering setup so the agent gets the intended
    # # medkit-based shaping signal instead of only the raw living reward.
    env = DoomGatheringRewardShaping(env)

    # grayscale and resize wrapper
    env = ImageTransformationWrapper(env, (84, 84))

    # glaucoma wrapper
    steps_until_decay = getattr(cfg, "steps_until_decay")
    decay_speed = getattr(cfg, "decay_speed")
    env = GlaucomaWrapper(env, steps_until_decay, decay_speed, 0.0)

    # calculate_agent_trajectory wrapper(good to see agent's trajectory while training)
    if calculate_agent_trajectory:
        trajectory_dir="play_yourself_images"
        if env_config is not None:
            worker_idx = env_config.worker_index
            env_idx = env_config.vector_index
            trajectory_dir = os.path.join(
                getattr(cfg, "train_dir"),
                getattr(cfg, "experiment"),
                "agent_trajectories",
                f"worker_{worker_idx}_env_{env_idx}",
            )
        aux_dir = getattr(cfg, "calculate_agent_trajectory_directory")
        if aux_dir is not None:
            trajectory_dir = os.path.join(
                getattr(cfg, "train_dir"),
                getattr(cfg, "experiment"),
                "agent_trajectories",
                aux_dir,
            )
        calculate_agent_trajectory_step = getattr(cfg, "calculate_agent_trajectory_step")
        env = TrajectoryVisualizationWrapper(env, trajectory_dir, calculate_agent_trajectory_step)

    return env

def register_custom_doom_env():
    register_env("custom_health_gathering", make_custom_env)
