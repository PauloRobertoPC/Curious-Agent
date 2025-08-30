import os
import sys
import argparse
from typing import Optional

# making the packages below visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sample_factory.enjoy import enjoy
from sample_factory.train import run_rl
from sample_factory.envs.env_utils import register_env
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args


from env.vizdoomenv import VizDoomGym
from wrappers.render_wrapper import RenderWrapper
from wrappers.glaucoma import GlaucomaWrapper
from wrappers.intrinsic_reward import IntrinsicRewardWrapper
from wrappers.image_transformation import ImageTransformationWrapper

from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

def make_custom_env(full_env_name: str, cfg=None, env_config=None, render_mode: Optional[str] = None):
    env = VizDoomGym(render_mode=render_mode)
    env = ImageTransformationWrapper(env, (161, 161))
    env = GlaucomaWrapper(env, 0, cfg["glaucoma_level"], -100)

    # env = IntrinsicRewardWrapper(env)
    
    # if render_mode == "rgb_array":
        # env = RenderWrapper(env)
        # env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)
    return env

def register_custom_env_envs():
    # register the env in sample-factory's global env registry
    # after this, you can use the env in the command line using --env=custom_env_name
    register_env("health_gathering_glaucoma", make_custom_env)

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue

def add_custom_env_args(_env, p: argparse.ArgumentParser, evaluation=False):
    # You can extend the command line arguments here
    # p.add_argument("--custom_argument", default="value", type=str, help="")
    # p.add_argument(
    #     "--action",
    #     default="play",
    #     choices=["train", "play"],
    #     type=str,
    #     help=(f'choices=["train", "play"]')
    # )
    p.add_argument(
        "--glaucoma_level",
        type=int,
        default=0,
        help="Glaucoma severity level (must be a positive integer)",
    )
    p.add_argument(
        "--reward_type",
        type=str,
        default="extrinsic",
        choices=["extrinsic", "rnd", "icm", "ngu"],
        help="Type of reward to use during training.",
    )

def parse_args(custom_env_override_defaults, argv=None, evaluation=False):
    # parse the command line arguments to build
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_custom_env_args(partial_cfg.env, parser, evaluation=evaluation)
    custom_env_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg
