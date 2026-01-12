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

class HealthGatheringCallback(BaseCallback):

    def __init__(self, check_freq:int, save_path:str, glaucoma_level:int, env_setup: EnvSetup, verbose:int = 0):
        super(HealthGatheringCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.buffer = None
        self.env_setup = env_setup

        self.random_len_ep = 0
        self.random_mean_len_ep = deque(maxlen=50)
        self.random_mean_max_glaucoma_len = deque(maxlen=50)
        self.random_mean_max_steps_with_hungry = deque(maxlen=50)
        self.random_mean_medkits_used = deque(maxlen=50)


        last_eval_layout = self.env_setup.info["eval_layout"] 
        self.eval_name = ["square", "circle", "sin", "grid"]
        self.eval_envs = []
        self.eval_len = []
        self.eval_max_steps_with_hungry = []
        self.eval_medkits_used = []
        for i in range(1, 5):
            self.env_setup.info["eval_layout"] = i
            self.eval_envs.append(self.env_setup.make_env())
            self.eval_len.append(deque(maxlen=50))
            self.eval_max_steps_with_hungry.append(deque(maxlen=50))
            self.eval_medkits_used.append(deque(maxlen=50))
        self.env_setup.info["eval_layout"] = last_eval_layout

    def _init_callback(self):
        self.buffer = self.model.rollout_buffer
            

    def _on_step(self) -> bool:
        self.random_len_ep += 4
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

            aux_model_path = os.path.join(self.save_path, f"aux")
            self.model.save(aux_model_path)
            
            model = PPO.load(aux_model_path)
            for i, env in enumerate(self.eval_envs):
                # playing
                len_ep, info = play_episode(env, model)
                # adding info
                self.eval_len[i].append(len_ep)
                self.eval_max_steps_with_hungry[i].append(info["max_steps_with_hungry"])
                self.eval_medkits_used[i].append(info["medkits_used"])
                # logging raw values
                self.logger.record(f"{self.eval_name[i]}/episode_len", len_ep)
                self.logger.record(f"{self.eval_name[i]}/max_steps_with_hungry", info["max_steps_with_hungry"])
                self.logger.record(f"{self.eval_name[i]}/medkits_used", info["medkits_used"])
                # logging mean values
                self.logger.record(f"{self.eval_name[i]}/mean_episode_len", np.mean(self.eval_len[i]))
                self.logger.record(f"{self.eval_name[i]}/mean_max_steps_with_hungry", np.mean(self.eval_max_steps_with_hungry[i]))
                self.logger.record(f"{self.eval_name[i]}/mean_medkits_used", np.mean(self.eval_medkits_used[i]))

            self.logger.dump(step=self.n_calls)
            os.remove(f"{aux_model_path}.zip")


        if self.n_calls%self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)

        return True
    
    def _on_rollout_end(self) -> None:
        pass


class HealthGatheringBase(EnvSetup):
    def __init__(self, info:Dict[str, Union[str, int]], save_trajectories_images:bool, cfg_file:str):
        if "reward" not in info or  "glaucoma_level" not in info or "rnd_strength" not in info or "render_mode" not in info:
            raise ValueError("It should contains 'reward', 'glaucoma_level', 'rnd_strength' and 'render_mode'")
        self.reward = info["reward"]
        self.glaucoma_level = info["glaucoma_level"]
        self.rnd_strength = info["rnd_strength"]
        self.info = info
        self.save_trajectories_images = save_trajectories_images
        if "eval_layout" not in self.info:
            self.info["eval_layout"] = 0
        self.eval_layout_to_name = ["random", "square", "circle", "sin", "grid"]
        super().__init__(cfg_file, 3, (3, 240, 320))
     
    def set_train_and_logging_callback(self):
        self.train_and_logging_callback = HealthGatheringCallback(CHECKPOINT_FREQUENCY, self.get_train_dir(), self.glaucoma_level, self)

    def get_info(self, state):
        self.last_medkits_used = int(state.game_variables[1])
        return  { "ammo": state.game_variables[0], "medkits_used": self.last_medkits_used }

    def zero_info(self):
        return  { "ammo": 0, "medkits_used": 0}

    def done_info(self, info):
        info["medkits_used"] = self.last_medkits_used
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
        return f"./train/hg_{self.reward}{self.rnd_strength}_{self.glaucoma_level}g"

    def _get_log_dir(self) -> str:
        return f"./logs/hg_{self.reward}{self.rnd_strength}_{self.glaucoma_level}g"

class HealthGathering(HealthGatheringBase):
    def __init__(self, info, save_trajectories_images):
        super().__init__(info, save_trajectories_images, "health_gathering.cfg")

class HealthGatheringNoLife(HealthGatheringBase):
    def __init__(self, info, save_trajectories_images):
        super().__init__(info, save_trajectories_images, "health_gathering_no_life.cfg")

    def _get_train_dir(self) -> str:
        return f"./train/hgnl_{self.reward}{self.rnd_strength}_{self.glaucoma_level}g"

    def _get_log_dir(self) -> str:
        return f"./logs/hgnl_{self.reward}{self.rnd_strength}_{self.glaucoma_level}g"