import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from ray.tune.registry import register_env
from game.computer_assembly_game_v2 import ComputerAssemblyGame
import pygame

def env_creator(env_config):
    return ComputerAssemblyMultiAgentEnv(env_config)

register_env("computer_assembly_multi_agent_env", env_creator)

class ComputerAssemblyMultiAgentEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config={}):
        super(ComputerAssemblyMultiAgentEnv, self).__init__()
        self.game = ComputerAssemblyGame()
        self.action_space = spaces.Discrete(11)  # Action space for each agent is 0-10
        self.agents = ["agent_1", "agent_2"]
        # Define the observation space based on the game environment
        n_components = 8
        obs_space_low = [0, 0, 1] * n_components + [0, 0, -1] * 2 + [0]
        obs_space_high = [self.game.screen_width / 100, self.game.screen_height / 100, 4] * n_components + [
            self.game.screen_width / 100, self.game.screen_height / 100, 9] * 2 + [1]
        # self.observation_space = spaces.Box(low=np.array(obs_space_low), high=np.array(obs_space_high),
        #                                     dtype=np.int32)
        discrete_screen_width_steps = 15  # Example: screen width divided into 8 discrete steps
        discrete_screen_height_steps = 10  # Example: screen height divided into 6 discrete steps
        other_component_steps = [4, 10, 1]  # Example: other observations with fixed discrete steps

        # Combine all steps into a single array for MultiDiscrete
        multi_discrete_obs_space = [discrete_screen_width_steps,
                                    discrete_screen_height_steps,4] * n_components + [discrete_screen_width_steps,
                                    discrete_screen_height_steps,10]*2 +[1]

        self.observation_space = {
            agent: spaces.MultiDiscrete(np.array(multi_discrete_obs_space) ) for agent in self.agents
        }

        self.current_step = 0
        self.max_steps = 10000
    def reset(self, seed=None, options=None):
        self.current_step = 0
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # Reset the game to its initial state
        self.game.reset_game(seed)  # Assuming reset_game doesn't need a seed. If it does, pass seed here.
        observation = self._get_observation()
        print(observation)
        # In a multi-agent environment, return observations for each agent
        return {"agent_1": observation, "agent_2": observation}, {}

    def step(self, action_dict):
        self.current_step += 1

        # Extract actions for each agent from the action dictionary
        action_agent_1 = action_dict["agent_1"]
        action_agent_2 = action_dict["agent_2"]

        # Simulate actions in the environment and update state
        reward, terminated = self.game.move_hands_simultaneously(action_agent_1, action_agent_2)

        observation = self._get_observation()
        done = terminated or self.current_step >= self.max_steps
        truncated = self.current_step >= self.max_steps
        # Provide observations, rewards, and done flags for each agent
        obs = {"agent_1": observation, "agent_2": observation}
        rewards = {"agent_1": reward, "agent_2": reward}
        dones = {"agent_1": done, "agent_2": done, "__all__": done}
        infos = {"agent_1": {}, "agent_2": {}}
        truncateds = {"agent_1": truncated, "agent_2": truncated}

        return obs, rewards, dones, truncateds, infos

    def render(self, mode='human'):
        self.game.render()

    def close(self):
        pygame.quit()

    def _get_observation(self):
        # Initialize an empty list to hold observation data
        obs = []

        # Loop through each component and append its position and state to the observation list
        for component_name, component_data in self.game.states.items():
            obs.append(int(component_data['position'][0])/100)  # x position, cast to int
            obs.append(int(component_data['position'][1])/100)  # y position, cast to int
            obs.append(int(component_data['state']))  # state, cast to int

        obs.append(self.game.hand_x/100)
        obs.append(self.game.hand_y/100)
        if self.game.action_hand1 is None:
            obs.append(-1)
        else:
            obs.append(self.game.action_hand1)
        obs.append(self.game.second_hand_x/100)
        obs.append(self.game.second_hand_y/100)
        if self.game.action_hand2 is None:
            obs.append(-1)
        else:
            obs.append(self.game.action_hand2)
        if self.game.hand2_waiting_for_handover:
            obs.append(1)
        else:
            obs.append(0)

        # Convert the observation list to a NumPy array with dtype=np.int32
        observation = np.array(obs, dtype=np.int32)

        return observation