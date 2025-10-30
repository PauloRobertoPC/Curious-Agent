import os
import sys
import time
import cv2
import torch
import numpy as np

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
from wrappers.intrinsic_reward import IntrinsicRewardWrapper
from wrappers.image_transformation import ImageTransformationWrapper

CHECKPOINT_DIR = "./train/health_gathering"
LOG_DIR = "./logs/log_health_gathering"
MODEL_NAME = f"./train/health_gathering/best_model_500000"

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
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        self.buffer = self.model.rollout_buffer
            

    def _on_step(self) -> bool:
        if self.n_calls%self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)
        return True
    
    def _on_rollout_end(self) -> None:
        pass

def make_env(render_mode=None):
    env = VizDoomGym(render_mode=render_mode)
    env = ImageTransformationWrapper(env, (161, 161))
    env = GlaucomaWrapper(env, 0, 150, -100)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # env = IntrinsicRewardWrapper(env)
    
    if render_mode == "rgb_array":
        env = RenderWrapper(env)
        env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)
    return env

def play(randomly=False):
    env = make_env(render_mode=None)
    model = None
    cam = None
    if not randomly:
        model = PPO.load(MODEL_NAME)
        cam = instantiate_cam(ModelCamWrapper(model.policy))
    
    episodes = 5
    for episode in range(episodes):
        total_reward = 0
        finished = False
        obs, _ = env.reset()
        while not finished:
            if randomly:
                action = env.action_space.sample()
                show_game(obs)
            else:
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
    
def evaluate():
    model = PPO.load(MODEL_NAME)
    
    env = make_env()

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    env.close()

    print(mean_reward)

def train():
    envs = make_vec_env(make_env, n_envs=1)
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    
    model = PPO("CnnPolicy", envs, tensorboard_log=LOG_DIR, learning_rate=0.0001, n_steps=4096, policy_kwargs=dict(normalize_images=False))
    
    model.learn(total_timesteps=2000000, callback=callback, progress_bar=True)
    
    envs.close()
    

def callback_playing(obs_t, obs_tp1, action, reward, terminated, truncated, info):
    # cv2.imwrite("output.png", np.moveaxis(obs_tp1, 0, -1))
    print(reward)
    # print(info)

def play_human():
    env = make_env(render_mode="rgb_array")
    playing(env, keys_to_action={ "a": 0, "d": 1, "w": 2 }, wait_on_player=True, callback=callback_playing)
    env.close()


if __name__ == "__main__":
    # train()
    # evaluate()
    play(False)
    # record()
    # play_human()
