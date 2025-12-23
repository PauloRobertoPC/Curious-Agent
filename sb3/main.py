import os
import sys
import time
import torch
import argparse
import numpy as np
from collections import deque

from cam import *
from utils import *

from gymnasium.utils.play import play as playing

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.vizdoomenv import VizDoomGym
from env.env_setup import EnvSetup, CHECKPOINT_FREQUENCY

envs = {}
from env.health_gathering import HealthGathering
envs["hg"] = HealthGathering

def train(es: EnvSetup):
    es.create_directory()
    es.set_train_and_logging_callback()
    save_experiment_info(es.get_train_dir(), es.info)
    envs = make_vec_env(es.make_env, n_envs=1)
    
    model = PPO("CnnPolicy", envs, tensorboard_log=es.get_log_dir(), learning_rate=0.0001, n_steps=4096, policy_kwargs=dict(normalize_images=False))
    
    model.learn(total_timesteps=2000000, callback=es.train_and_logging_callback, progress_bar=True)
    
    envs.close()

def play(es: EnvSetup, model_name:str):
    env = es.make_env()
    model = PPO.load(model_name)
    cam = instantiate_cam(ModelCamWrapper(model.policy))
    
    episodes = 5
    for episode in range(episodes):
        total_reward = 0
        finished = False
        obs, _ = env.reset()
        while not finished:
            action, _ = model.predict(obs)
            show_game(obs, cam, [ClassifierOutputTarget(action)])
            obs, reward, done, truncated, info = env.step(action)
            time.sleep(0.05)
            total_reward += reward
            finished = done or truncated
        print(f"Total Reward for episode {episode} is {total_reward}.")
        time.sleep(2)

    env.close()

def evaluate(es: EnvSetup, model_name:str, eval_episodes:int):
    env = es.make_env()

    model = PPO.load(model_name)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=eval_episodes)

    env.close()

    print(mean_reward)

def play_human(es: EnvSetup):
    env = VizDoomGym(es)
    key_to_action = {
        "a": 0,
        "d": 1,
        "w": 2
    }
    playing(env, keys_to_action=key_to_action, wait_on_player=True)
    env.close()

# TODO:
def record(eval_layout):
    env = make_env(eval_layout=eval_layout, render_mode="rgb_array")
    model = PPO.load(MODEL_NAME)
    
    episodes = 5
    for episode in range(episodes):
        total_reward = 0
        finished = False
        obs, _ = env.reset()
        while not finished:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            finished = done or truncated
        print(f"Total Reward for episode {episode} is {total_reward}.")

    env.close()
    
def experiments(env:str):
    exps = [
        {"glaucoma_level": 0, "reward": "extrinsic", "rnd_strength": 0, "render_mode": None},
    ]
    for e in exps:
        es = envs[env]({**e})
        train(es)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinforcement Learning with Stables Baseline 3")
    parser.add_argument("--action", required=True, type=str, choices=["train", "play", "evaluate", "debug", "experiment"], help="'train' the agent, watch it 'play','evaluate' its rewards, 'debug' the unwraped enviroment or perform 'experiment' as in experiments function")
    parser.add_argument("--env", required=True, type=str, choices=["hg", "hgs"], help="enviroment to be use, hg=health_gathering, hgs=health_gathering_supreme")
    parser.add_argument("--experiment", type=str, help="experiment name, all the files regarding this experiment will be saved in train/<experiment>")
    parser.add_argument("--reward", type=str, choices=["rnd", "extrinsic"], help="reward given to the agent(required in train)")
    parser.add_argument("--glaucoma_level", type=int, help="glaucoma growth intensity(required in train)")
    parser.add_argument("--model", type=int, help=f"the model number you wanna se the agent using as the policy(it should be multiple of {CHECKPOINT_FREQUENCY})(required in play and evaluate)")
    parser.add_argument("--eval_episodes", type=int, default=10, help="total of episodes you want to evaluate the reward(used in evaluate)")
    parser.add_argument("--layout", type=int, help="layout to be played 0 - random, 1 - square, 2 - circle, 3 - sine curve, 4 - grid")
    parser.add_argument("--rnd_strength", type=int, help="multiplier for the intrinsic reward calculated by the rnd")

    args = parser.parse_args()

    if args.action == "train":
        # checking if argument were passed
        check_reward(parser, args)
        check_glaucoma_level(parser, args)
        st = 0
        if args.reward == "rnd":
            check_rnd_strength(parser, args)
            st = args.rnd_strength
        # train itself
        es = envs[args.env]({
            "glaucoma_level": args.glaucoma_level,
            "reward": args.reward,
            "rnd_strength": st,
            "render_mode": None,
        })
        train(es)
    elif args.action == "play":
        check_experiment(parser, args)
        checkpoint_dir = f"./train/{args.experiment}"
        model_name = check_experiment_on_play(parser, args, checkpoint_dir)
        check_layout(parser, args)
        config = read_experiment_info(checkpoint_dir)
        config["reward"] = "extrinsic"
        es = envs[args.env]({
            "glaucoma_level": config["glaucoma_level"],
            "reward": config["reward"],
            "rnd_strength": config["rnd_strength"],
            "render_mode": None,
            "eval_layout": args.layout
        })
        play(es, model_name)
    elif args.action == "evaluate":
        check_experiment(parser, args)
        checkpoint_dir = f"./train/{args.experiment}"
        model_name = check_experiment_on_play(parser, args, checkpoint_dir)
        check_layout(parser, args)
        config = read_experiment_info(checkpoint_dir)
        config["reward"] = "extrinsic"
        es = envs[args.env]({
            "glaucoma_level": config["glaucoma_level"],
            "reward": config["reward"],
            "rnd_strength": config["rnd_strength"],
            "render_mode": None,
            "eval_layout": args.layout
        })
        evaluate(es, model_name, args.eval_episodes)
    elif args.action == "debug":
        check_layout(parser, args)
        es = envs[args.env]({
            "glaucoma_level": 0,
            "reward": "extrinsic",
            "rnd_strength": 0,
            "render_mode": "rgb_array",
            "eval_layout": args.layout
        })
        play_human(es)
    elif args.action == "experiment":
        experiments(args.env)