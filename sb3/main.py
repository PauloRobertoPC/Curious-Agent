import os
import sys
import time
import cv2
import torch
import argparse
import numpy as np

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

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq:int, save_path:str, verbose:int = 0):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.buffer = None

    def _init_callback(self):
        self.buffer = self.model.rollout_buffer
            

    def _on_step(self) -> bool:
        if "max_glaucoma_len" in self.locals["infos"][0]:
            self.logger.record("glaucoma/max_glaucoma_len", self.locals["infos"][0]["max_glaucoma_len"])
            self.logger.record("glaucoma/max_steps_with_hungry", self.locals["infos"][0]["max_steps_with_hungry"])
        if self.n_calls%self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)
        return True
    
    def _on_rollout_end(self) -> None:
        pass

def make_env(glaucoma_level:int, reward:str, render_mode=None):
    print(f"ENV CONFIG -> {locals()}")
    env = VizDoomGym(render_mode=render_mode)
    env = ImageTransformationWrapper(env, (161, 161))
    env = GlaucomaWrapper(env, 0, glaucoma_level, -100)
    
    if reward == "rnd":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = RNDWrapper(env, device=device)
    
    if render_mode == "rgb_array":
        env = RenderWrapper(env)
        env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)
    return env

def play():
    config = read_experiment_info(CHECKPOINT_DIR)
    config["reward"] = "extrinsic"
    env = make_env(render_mode=None, **config)
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

def record():
    env = make_env(render_mode="rgb_array")
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
    
def evaluate(eval_episodes):
    config = read_experiment_info(CHECKPOINT_DIR)
    config["reward"] = "extrinsic"
    env = make_env(render_mode=None, **config)

    model = PPO.load(MODEL_NAME)
    
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=eval_episodes)

    env.close()

    print(mean_reward)

def train(glaucoma_level:int, reward:str):
    save_experiment_info(CHECKPOINT_DIR, locals())
    envs = make_vec_env(make_env, n_envs=1, env_kwargs=locals())
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    
    model = PPO("CnnPolicy", envs, tensorboard_log=LOG_DIR, learning_rate=0.0001, n_steps=4096, policy_kwargs=dict(normalize_images=False))
    
    model.learn(total_timesteps=2000000, callback=callback, progress_bar=True)
    
    envs.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinforcement Learning with Stables Baseline 3")
    parser.add_argument("--action", required=True, type=str, choices=["train", "play", "evaluate"], help="'train' the agent, watch it 'play' or 'evaluate' its rewards")
    parser.add_argument("--experiment", required=True, type=str, help="experiment name, all the files regarding this experiment will be saved in train/<experiment>")
    parser.add_argument("--reward", type=str, choices=["rnd", "extrinsic"], help="reward given to the agent(required in train)")
    parser.add_argument("--glaucoma_level", type=int, help="glaucoma growth intensity(required in train)")
    parser.add_argument("--model", type=int, help="the model number you wanna se the agent using as the policy(it should be multiple of 10000)(required in play and evaluate)")
    parser.add_argument("--eval_episodes", type=int, default=10, help="total of episodes you want to evaluate the reward(used in evaluate)")

    args = parser.parse_args()

    CHECKPOINT_DIR = f"./train/{args.experiment}"
    LOG_DIR = f"./logs/{args.experiment}"

    if args.action == "train":
        # deciding what to do if the experiment already exists
        check_experiment_on_train(args, CHECKPOINT_DIR, LOG_DIR)
        # checking if argument were passed
        check_reward(parser, args)
        check_glaucoma_level(parser, args)
        # train itself
        train(args.glaucoma_level, args.reward)
    elif args.action == "play":
        MODEL_NAME = check_experiment_on_play(parser, args, CHECKPOINT_DIR)
        play()
    elif args.action == "evaluate":
        MODEL_NAME = check_experiment_on_play(parser, args, CHECKPOINT_DIR)
        evaluate(args.eval_episodes)
    # # record()