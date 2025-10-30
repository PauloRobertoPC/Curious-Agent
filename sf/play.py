import os
import sys

from utils import register_custom_env_envs, parse_args

# making the packages below visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sample_factory.enjoy import enjoy

def custom_env_override_defaults(_env, parser):
    parser.set_defaults(
        #experiment="doom_health_gathering/glaucoma_/01_glaucoma_see_1111_u.rnn_True_b.siz_2048_g.lev_50_r.typ_extrinsic_env_health_gathering_glaucoma"
        save_video=True,
        max_num_episodes=2,
        experiment="glaucoma150"
    )

if __name__ == "__main__":
    register_custom_env_envs()
    cfg = parse_args(custom_env_override_defaults, evaluation=True)
    status = enjoy(cfg)
    sys.exit(status)
