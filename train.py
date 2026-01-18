from rllte.agent import PPO
from rllte.env import health_gathering
from rllte.xplore.reward import RND
from wrappers import image_wrapper, glaucoma_wrapper
from gymnasium.wrappers import TransformReward

def transform_reward_wrapper(f):
    return lambda env: TransformReward(env, f)

if __name__ == "__main__":
    # env setup
    device = "cuda:0"
    wrappers = [
        image_wrapper((84, 84)),
        # glaucoma_wrapper(0, ),
        transform_reward_wrapper(lambda r: 0)
        
    ]
    env = health_gathering(num_envs=8, device=device, asynchronous=False, seed=1, wrappers=wrappers)
    eval_env = health_gathering(num_envs=8, device=device, asynchronous=False, seed=9, wrappers=wrappers)
    # create agent
    agent = PPO(env=env, 
                eval_env=eval_env, 
                device=device,
                tag="health_gathering_0g_rnd",
                )
     # create intrinsic reward
    rnd = RND(envs=env, device=device)
    # set the module
    agent.set(reward=rnd)
    # start training
    agent.train(num_train_steps=8_000_000, save_interval=1)