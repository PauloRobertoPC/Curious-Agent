import sys

# Sample Factory Imports
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.enjoy import enjoy
from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components

from .utils import add_custom_args, register_custom_doom_env

def main():
    """
    Script dedicado para visualizar (enjoy) o ambiente da Tese.
    """
    register_vizdoom_components()

    register_custom_doom_env()

    parser, cfg = parse_sf_args(evaluation=True)

    add_custom_args(parser)

    cfg = parse_full_cfg(parser)

    status = enjoy(cfg)
    return status

if __name__ == "__main__":
    sys.exit(main())
