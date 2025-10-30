import os
import sys

from utils import register_custom_env_envs, parse_args

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sample_factory.train import run_rl

def custom_env_override_defaults(_env, parser):
    # Modify the default arguments when using this env.
    # These can still be changed from the command line. See configuration guide for more details.
    parser.set_defaults(
        gamma=0.99,
        learning_rate=1e-4,
        lr_schedule="constant",
        adam_eps=1e-5,  
        train_for_env_steps=100_000_000,
        algo="APPO",
        env_frameskip=4,
        use_rnn=False,
        batched_sampling=True,
        batch_size=2048, 
        num_workers=8, 
        num_envs_per_worker=16, 
        device="gpu",
        num_policies=1,
        experiment="glaucoma150_more_steps",
        glaucoma_level=150,
        reward_type="extrinsic",
    )

if __name__ == "__main__":
    register_custom_env_envs()
    cfg = parse_args(custom_env_override_defaults)
    status = run_rl(cfg)
    sys.exit(status)