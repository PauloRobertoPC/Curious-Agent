import os
import sys
import time
import cv2
import torch
import argparse
import numpy as np
from collections import deque

from utils import *

from gymnasium.wrappers import RecordVideo
from gymnasium.utils.play import play as playing

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# making the packages below visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.vizdoomenv import VizDoomGym
from wrappers.render_wrapper import RenderWrapper
from wrappers.glaucoma import GlaucomaWrapper
from wrappers.image_transformation import ImageTransformationWrapper
from wrappers.rnd_wrapper import RNDWrapper

CHECKPOINT_DIR = ""
LOG_DIR = ""
MODEL_NAME = ""

class ModelCamWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(ModelCamWrapper, self).__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.extract_features(x)
        if self.model.share_features_extractor:
            latent_pi, latent_vf = self.model.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.model.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.model.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        distribution = self.model._get_action_dist_from_latent(latent_pi)
        return distribution.distribution.logits

def instantiate_cam(model):
    target_layers = [
        # model.model.features_extractor.cnn[-7],
        # model.model.features_extractor.cnn[-6],
        # model.model.features_extractor.cnn[-5],
        # model.model.features_extractor.cnn[-4],
        model.model.features_extractor.cnn[-3],
        # model.model.features_extractor.cnn[-2],
        # model.model.features_extractor.cnn[-1], DON't USE THIS ONE
    ]

    # You can choose different CAM methods here
    cam = GradCAM(model=model, target_layers=target_layers)
    # cam = HiResCAM(model=model, target_layers=target_layers)
    # cam = ScoreCAM(model=model, target_layers=target_layers)
    # cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    # cam = AblationCAM(model=model, target_layers=target_layers)
    # cam = XGradCAM(model=model, target_layers=target_layers)
    # cam = EigenCAM(model=model, target_layers=target_layers)
    # cam = FullGrad(model=model, target_layers=target_layers)
    return cam

# @torch.enable_grad()
def generate_cam(cam, img, targets):
    img_batch = torch.from_numpy(img).unsqueeze(0) #.to("cuda")
    grayscale_cam = cam(input_tensor=img_batch, targets=targets, aug_smooth=False, eigen_smooth=False)
    grayscale_cam = grayscale_cam[0, :] # since we have a single image in the batch

    img_hwc = np.transpose(img, (1, 2, 0))
    rgb_image = np.repeat(img_hwc, 3, axis=2)
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return visualization

def show_game(obs, cam=None, targets=None):
    img_hwc = np.transpose(obs, (1, 2, 0))
    image = (np.repeat(img_hwc, 3, axis=2)*255).astype(np.uint8)
    if cam is not None:
        cam_image = generate_cam(cam, obs, targets)
        image = np.concatenate((image, cam_image), axis=1)
    image_rgb = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    image_bgr = image_rgb[:, :, ::-1]  # convert RGB to BGR for OpenCV
    cv2.imshow("Agent Playing", image_bgr)
    cv2.waitKey(1)

def make_env(glaucoma_level:int, reward:str, eval_layout:int, strength=0, render_mode=None):
    print(f"ENV CONFIG -> {locals()}")
    env = VizDoomGym(eval_layout=eval_layout, render_mode=render_mode)
    env = ImageTransformationWrapper(env, (161, 161))
    env = GlaucomaWrapper(env, 0, glaucoma_level, -100)
    
    if reward == "rnd":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = RNDWrapper(env=env, rnd_strength=strength, device=device)
    
    if render_mode == "rgb_array":
        env = RenderWrapper(env)
        env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)
    return env

def play_episode(env, model):
    total_reward = 0
    finished = False
    info = {}
    obs, _ = env.reset()
    ep_len = 0
    while not finished:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        ep_len += 1
        finished = done or truncated
    return ep_len, info

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq:int, save_path:str, glaucoma_level:int, verbose:int = 0):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.buffer = None

        self.random_len_ep = 0
        self.random_mean_len_ep = deque(maxlen=50)
        self.random_mean_max_glaucoma_len = deque(maxlen=50)
        self.random_mean_max_steps_with_hungry = deque(maxlen=50)

        self.eval_name = ["square", "circle", "sin", "grid"]
        self.eval_envs = []
        self.eval_len = []
        self.eval_max_glaucoma_len = []
        self.eval_max_steps_with_hungry = []
        for i in range(1, 5):
            self.eval_envs.append(make_env(glaucoma_level=glaucoma_level, reward="extrinsic", eval_layout=i))
            self.eval_len.append(deque(maxlen=50))
            self.eval_max_glaucoma_len.append(deque(maxlen=50))
            self.eval_max_steps_with_hungry.append(deque(maxlen=50))

    def _init_callback(self):
        self.buffer = self.model.rollout_buffer
            

    def _on_step(self) -> bool:
        self.random_len_ep += 1
        if "max_glaucoma_len" in self.locals["infos"][0]:

            # adding info
            self.random_mean_len_ep.append(self.random_len_ep)
            self.random_mean_max_glaucoma_len.append(self.locals["infos"][0]["max_glaucoma_len"])
            self.random_mean_max_steps_with_hungry.append(self.locals["infos"][0]["max_steps_with_hungry"])
            # logging raw values
            self.logger.record(f"glaucoma/random/episode_len", self.random_len_ep)
            self.logger.record(f"glaucoma/random/max_glaucoma_len", self.locals["infos"][0]["max_glaucoma_len"])
            self.logger.record(f"glaucoma/random/max_steps_with_hungry", self.locals["infos"][0]["max_steps_with_hungry"])
            # logging mean values
            self.logger.record(f"glaucoma/random/mean_episode_len", np.mean(self.random_len_ep))
            self.logger.record(f"glaucoma/random/mean_max_glaucoma_len", np.mean(self.random_mean_max_glaucoma_len))
            self.logger.record(f"glaucoma/random/mean_max_steps_with_hungry", np.mean(self.random_mean_max_steps_with_hungry))

            self.random_len_ep = 0

            aux_model_path = os.path.join(self.save_path, f"aux")
            self.model.save(aux_model_path)
            
            model = PPO.load(aux_model_path)
            for i, env in enumerate(self.eval_envs):
                # playing
                len_ep, info = play_episode(env, model)
                # adding info
                self.eval_len[i].append(len_ep)
                self.eval_max_glaucoma_len[i].append(info["max_glaucoma_len"])
                self.eval_max_steps_with_hungry[i].append(info["max_steps_with_hungry"])
                # logging raw values
                self.logger.record(f"glaucoma/{self.eval_name[i]}/episode_len", len_ep)
                self.logger.record(f"glaucoma/{self.eval_name[i]}/max_glaucoma_len", info["max_glaucoma_len"])
                self.logger.record(f"glaucoma/{self.eval_name[i]}/max_steps_with_hungry", info["max_steps_with_hungry"])
                # logging mean values
                self.logger.record(f"glaucoma/{self.eval_name[i]}/mean_episode_len", np.mean(self.eval_len[i]))
                self.logger.record(f"glaucoma/{self.eval_name[i]}/mean_max_glaucoma_len", np.mean(self.eval_max_glaucoma_len[i]))
                self.logger.record(f"glaucoma/{self.eval_name[i]}/mean_max_steps_with_hungry", np.mean(self.eval_max_steps_with_hungry[i]))

            os.remove(f"{aux_model_path}.zip")


        if self.n_calls%self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)

        return True
    
    def _on_rollout_end(self) -> None:
        pass

def play(eval_layout:int):
    config = read_experiment_info(CHECKPOINT_DIR)
    config["reward"] = "extrinsic"
    env = make_env(eval_layout=eval_layout, render_mode=None, **config)
    model = PPO.load(MODEL_NAME)
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
    
def evaluate(eval_episodes:int, eval_layout:int):
    config = read_experiment_info(CHECKPOINT_DIR)
    config["reward"] = "extrinsic"
    env = make_env(eval_layout=eval_layout, render_mode=None, **config)

    model = PPO.load(MODEL_NAME)
    
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=eval_episodes)

    env.close()

    print(mean_reward)

def train(glaucoma_level:int, reward:str, strength:int):
    save_experiment_info(CHECKPOINT_DIR, locals())
    eval_layout = 0
    print(locals())
    envs = make_vec_env(make_env, n_envs=1, env_kwargs=locals())
    callback = TrainAndLoggingCallback(check_freq=CHECKPOINT_FREQUENCY, save_path=CHECKPOINT_DIR, glaucoma_level=glaucoma_level)
    
    model = PPO("CnnPolicy", envs, tensorboard_log=LOG_DIR, learning_rate=0.0001, n_steps=4096, policy_kwargs=dict(normalize_images=False))
    
    model.learn(total_timesteps=2000000, callback=callback, progress_bar=True)
    
    envs.close()

def play_human(eval_layout:int):
    env = VizDoomGym(eval_layout=eval_layout, render_mode="rgb_array")
    key_to_action = {
        "a": 0,
        "d": 1,
        "w": 2
    }
    playing(env, keys_to_action=key_to_action, wait_on_player=True)
    env.close()

def experiments():
    rewards = [("rnd", 10000), ("extrinsic", 0), ("rnd", 1000), ("rnd", 100000)]
    glaucoma_levels = [50, 100, 150, 200, 250, 300]
    for reward in rewards:
        for glaucoma_level in glaucoma_levels:
            experiment = f"{reward[0]}{reward[1]}_{glaucoma_level}g"
            print(experiment)
            CHECKPOINT_DIR = f"./train/{experiment}"
            LOG_DIR = f"./logs/{experiment}"
            overwrite_experiment_on_train(CHECKPOINT_DIR, LOG_DIR)
            train(glaucoma_level, reward[0], reward[1])


if __name__ == "__main__":
    print(CHECKPOINT_FREQUENCY)
    parser = argparse.ArgumentParser(description="Reinforcement Learning with Stables Baseline 3")
    parser.add_argument("--action", required=True, type=str, choices=["train", "play", "evaluate", "debug", "experiment"], help="'train' the agent, watch it 'play','evaluate' its rewards, 'debug' the unwraped enviroment or perform 'experiment' as in experiments function")
    parser.add_argument("--experiment", type=str, help="experiment name, all the files regarding this experiment will be saved in train/<experiment>")
    parser.add_argument("--reward", type=str, choices=["rnd", "extrinsic"], help="reward given to the agent(required in train)")
    parser.add_argument("--glaucoma_level", type=int, help="glaucoma growth intensity(required in train)")
    parser.add_argument("--model", type=int, help=f"the model number you wanna se the agent using as the policy(it should be multiple of {CHECKPOINT_FREQUENCY})(required in play and evaluate)")
    parser.add_argument("--eval_episodes", type=int, default=10, help="total of episodes you want to evaluate the reward(used in evaluate)")
    parser.add_argument("--layout", type=int, help="layout to be played 0 - random, 1 - square, 2 - circle, 3 - sine curve, 4 - grid")
    parser.add_argument("--rnd_strength", type=int, help="multiplier for the intrinsic reward calculated by the rnd")

    args = parser.parse_args()

    CHECKPOINT_DIR = f"./train/{args.experiment}"
    LOG_DIR = f"./logs/{args.experiment}"

    if args.action == "train":
        # deciding what to do if the experiment already exists
        check_experiment_on_train(args, CHECKPOINT_DIR, LOG_DIR)
        # checking if argument were passed
        check_reward(parser, args)
        check_glaucoma_level(parser, args)
        st = 0
        if args.reward == "rnd":
            check_rnd_strength(parser, args)
            st = args.rnd_strength
        # train itself
        train(args.glaucoma_level, args.reward, st)
    elif args.action == "play":
        MODEL_NAME = check_experiment_on_play(parser, args, CHECKPOINT_DIR)
        check_layout(parser, args)
        play(args.layout)
    elif args.action == "evaluate":
        MODEL_NAME = check_experiment_on_play(parser, args, CHECKPOINT_DIR)
        check_layout(parser, args)
        evaluate(eval_episodes=args.eval_episodes, eval_layout=args.layout)
    elif args.action == "debug":
        check_layout(parser, args)
        play_human(args.layout)
    elif args.action == "experiment":
        experiments()