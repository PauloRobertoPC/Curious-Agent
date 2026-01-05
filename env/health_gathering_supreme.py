from collections import deque
import os
import sys
import numpy as np
from typing import Dict, Sequence, Union
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.env_setup import EnvSetup, play_episode, CHECKPOINT_FREQUENCY
from env.vizdoomenv import VizDoomGym
from wrappers.glaucoma import GlaucomaWrapper
from wrappers.image_transformation import ImageTransformationWrapper
from wrappers.render_wrapper import RenderWrapper
from wrappers.rnd_wrapper import RNDWrapper
from gymnasium.wrappers import RecordVideo
from wrappers.trajectory_visualization import TrajectoryVisualizationWrapper

class HealthGatheringSupremeCallback(BaseCallback):

    def __init__(self, check_freq:int, save_path:str, glaucoma_level:int, env_setup: EnvSetup, verbose:int = 0):
        super(HealthGatheringSupremeCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.buffer = None
        self.env_setup = env_setup

        self.random_len_ep = 0
        self.random_mean_len_ep = deque(maxlen=50)
        self.random_mean_max_steps_with_hungry = deque(maxlen=50)
        self.random_mean_medkits_used = deque(maxlen=50)

    def _init_callback(self):
        self.buffer = self.model.rollout_buffer
            

    def _on_step(self) -> bool:
        self.random_len_ep += 1
        if "max_glaucoma_len" in self.locals["infos"][0]:

            # adding info
            self.random_mean_len_ep.append(self.random_len_ep)
            self.random_mean_max_steps_with_hungry.append(self.locals["infos"][0]["max_steps_with_hungry"])
            self.random_mean_medkits_used.append(self.locals["infos"][0]["medkits_used"])
            # logging raw values
            self.logger.record(f"random/episode_len", self.random_len_ep)
            self.logger.record(f"random/max_steps_with_hungry", self.locals["infos"][0]["max_steps_with_hungry"])
            self.logger.record(f"random/medkits_used", self.locals["infos"][0]["medkits_used"])

            # logging mean values
            self.logger.record(f"random/mean_episode_len", np.mean(self.random_len_ep))
            self.logger.record(f"random/mean_max_steps_with_hungry", np.mean(self.random_mean_max_steps_with_hungry))
            self.logger.record(f"random/mean_medkits_used", np.mean(self.random_mean_medkits_used))

            self.random_len_ep = 0
            self.logger.dump(step=self.n_calls)


        if self.n_calls%self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)

        return True
    
    def _on_rollout_end(self) -> None:
        pass


class HealthGatheringSupreme(EnvSetup):
    def __init__(self, info:Dict[str, Union[str, int]], save_trajectories_images:bool = True):
        if "reward" not in info or  "glaucoma_level" not in info or "rnd_strength" not in info or "render_mode" not in info:
            raise ValueError("It should contains 'reward', 'glaucoma_level', 'rnd_strength' and 'render_mode'")
        self.reward = info["reward"]
        self.glaucoma_level = info["glaucoma_level"]
        self.rnd_strength = info["rnd_strength"]
        self.info = info
        self.save_trajectories_images = save_trajectories_images
        if "eval_layout" not in self.info:
            self.info["eval_layout"] = 0
        super().__init__("health_gathering_supreme.cfg", 3, (3, 240, 320))
     
    def set_train_and_logging_callback(self):
        self.train_and_logging_callback = HealthGatheringSupremeCallback(CHECKPOINT_FREQUENCY, self.get_log_dir(), self.glaucoma_level, self)

    def get_info(self, state):
        self.last_medkits_used = int(state.game_variables[1])
        self.last_poisons_used = int(state.game_variables[2])
        return  { "ammo": state.game_variables[0], "medkits_used": self.last_medkits_used, "poisons_used": self.last_poisons_used }

    def zero_info(self):
        return  { "ammo": 0, "medkits_used": 0, "poisons_used": 0}

    def done_info(self, info):
        info["medkits_used"] = self.last_medkits_used
        info["poisons_used"] = self.last_poisons_used
        return info

    def make_env(self):
        print(f"ENV CONFIG -> {self.info}")
        env = VizDoomGym(self)
        if self.save_trajectories_images:
            env = TrajectoryVisualizationWrapper(env, f"{self.get_log_dir()}/{self.eval_layout_to_name[self.info["eval_layout"]]}_trajectories")
        env = ImageTransformationWrapper(env, (161, 161))
        env = GlaucomaWrapper(env, 0, self.info["glaucoma_level"], -100)
        
        if self.info["reward"] == "rnd":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            env = RNDWrapper(env=env, rnd_strength=self.info["rnd_strength"], device=device)
        
        if self.info["render_mode"] == "rgb_array":
            env = RenderWrapper(env)
            env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)
        return env

    def _get_train_dir(self) -> str:
        return f"./train/hgs_{self.reward}{self.rnd_strength}_{self.glaucoma_level}g"

    def _get_log_dir(self) -> str:
        return f"./logs/hgs_{self.reward}{self.rnd_strength}_{self.glaucoma_level}g"