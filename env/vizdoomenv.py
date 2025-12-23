from typing import Dict, Union
import gymnasium as gym
from vizdoom import *
from gymnasium.spaces import Discrete, Box, Sequence
from gymnasium import Env, make
import numpy as np

from env.env_setup import EnvSetup

class VizDoomGym(Env):
    metadata = { "render_modes": ["human", "rgb_array"], "render_fps": 30 }

    def __init__(self, env_setup:EnvSetup) -> None:
        super().__init__()

        self.env_setup = env_setup
        
        self.game = DoomGame()
        # choosing layout
        self.game.add_game_args(f"+set eval_layout {env_setup.info["eval_layout"]} +set tics_to_spawn_after_eat 120")
        self.game.load_config(env_setup.scenario)

        render_mode = env_setup.info["render_mode"]
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if render_mode != "human":
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
        self.game.init()

        # set of actions we can take in the enviroment
        self.actions = np.identity(env_setup.tot_actions, dtype=np.uint8)

        # spaces
        self.observation_space = Box(low=0, high=255, shape=env_setup.obs_space_shape, dtype=np.uint8)
        self.action_space = Discrete(env_setup.tot_actions)

    def reset(self, seed=None, options=None):
        self.game.new_episode()
        state = self.game.get_state()
        return state.screen_buffer, self.env_setup.get_info(state)
    
    def step(self, action):
        # doing step
        reward = self.game.make_action(self.actions[action], 4)
        
        # getting the other information
        state = self.game.get_state()
        if self.game.get_state():
            img = state.screen_buffer
            info = self.env_setup.get_info(state)
        else:
            img = np.zeros(self.observation_space.shape, dtype=np.uint8)
            info = self.env_setup.zero_info()


        done = self.game.is_episode_finished()
        if done:
            info = self.env_setup.done_info(info)
        
        return img, reward, done, False, info

    def render(self):
        state = self.game.get_state()
        if state:
            img = state.screen_buffer
        else:
            img = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return np.moveaxis(img, 0, -1)

    def close(self):
        self.game.close()