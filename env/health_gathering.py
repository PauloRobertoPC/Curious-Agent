from collections import deque
import os
import sys
import numpy as np
from typing import Dict, Sequence, Union
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
        print(self.info)
        self.eval_layout_to_name = ["random", "square", "circle", "sin", "grid"]
        super().__init__(cfg_file, 3, (3, 240, 320))
     
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
        # if self.save_trajectories_images:
        #     env = TrajectoryVisualizationWrapper(env, f"{self.get_log_dir()}/{self.eval_layout_to_name[self.info["eval_layout"]]}_trajectories")
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