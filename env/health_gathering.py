import os
import sys
import torch
import numpy as np
from typing import Dict, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.env_setup import EnvSetup
from env.vizdoomenv import VizDoomGym
from wrappers.glaucoma import GlaucomaWrapper
from wrappers.image_transformation import ImageTransformationWrapper
from wrappers.trajectory_visualization import TrajectoryVisualizationWrapper

class HealthGatheringBase(EnvSetup):
    def __init__(self, info:Dict[str, Union[str, int]], cfg_file:str, save_trajectories_images:str=""):
        if "glaucoma_level" not in info or "render_mode" not in info or "eval_layout" not in info:
            raise ValueError("It should contains 'reward', 'glaucoma_level' and 'render_mode'")
        self.glaucoma_level = info["glaucoma_level"]
        self.render_mode = info["render_mode"]
        self.eval_layout = info["eval_layout"]
        self.info = info
        self.save_trajectories_images = save_trajectories_images
        self.eval_layout_to_name = ["random", "square", "circle", "sin", "grid"]
        super().__init__(cfg_file, 6, (3, 240, 320))
     
    def get_info(self, state):
        self.last_medkits_used = int(state.game_variables[1])
        return  { "ammo": state.game_variables[0], "medkits_used": self.last_medkits_used }

    def zero_info(self):
        return  { "ammo": 0, "medkits_used": 0}

    def done_info(self, info):
        info["medkits_used"] = self.last_medkits_used
        return info

    def make_env(self):
        # print(f"ENV CONFIG -> {self.info}")
        env = VizDoomGym(self)
        if self.save_trajectories_images != "":
            env = TrajectoryVisualizationWrapper(env, self.save_trajectories_images)
        env = ImageTransformationWrapper(env, (84, 84))
        if self.glaucoma_level > 0:
            env = GlaucomaWrapper(env, 0, self.glaucoma_level, -100)
        return env

class HealthGathering(HealthGatheringBase):
    def __init__(self, info, save_trajectories_images:str):
        super().__init__(info, "health_gathering.cfg", save_trajectories_images)

class HealthGatheringNoLife(HealthGatheringBase):
    def __init__(self, info, save_trajectories_images:str):
        super().__init__(info, "health_gathering_no_life.cfg", save_trajectories_images)