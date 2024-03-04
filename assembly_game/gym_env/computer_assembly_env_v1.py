import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from game.computer_assembly_game_v1 import ComputerAssemblyGame
from gymnasium.envs.registration import register

register(
    id='ComputerAssemblyEnv-v1',  # Use an environment ID with a version number
    entry_point='gym_env.computer_assembly_env_v1:ComputerAssemblyEnv',  # Module path : ClassName
)

max_steps = 10000  # Define a maximum number of steps
current_step = 0  # Track the current step of the episode

class ComputerAssemblyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ComputerAssemblyEnv, self).__init__()
        self.game = ComputerAssemblyGame()
        self.action_space = spaces.MultiDiscrete([11, 11])  # Each hand can perform 9 actions
        # Each component has 2 position values (x, y) and 1 state value
        # Positions range from 0 to screen_width and 0 to screen_height
        # State ranges from 1 to 4 (or whatever your maximum state value is)
        # the components states, the hand1 and hand2 states, and the state of hand2 waiting for handover
        n_components = 8
        obs_space_low = [0, 0, 1] * n_components + [0,0,-1]*2 +[0]
        obs_space_high = [self.game.screen_width/100, self.game.screen_height/100, 4] * n_components +[self.game.screen_width/100, self.game.screen_height/100,9]*2+[1]
        self.observation_space = spaces.Box(low=np.array(obs_space_low), high=np.array(obs_space_high), dtype=np.int32)
        # n_components = 8  # Or however many components you have
        # obs_space_low = [1] * n_components +[0]
        # obs_space_high = [4] * n_components +[1]
        # self.observation_space = spaces.Box(low=np.array(obs_space_low), high=np.array(obs_space_high), dtype=np.int32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # Reset the game to its initial state
        self.game.reset_game(seed)  # Assuming reset_game doesn't need a seed. If it does, pass seed here.
        observation = self._get_observation()  # Get the initial game state as observation
        info = {}
        return observation, info



    def step(self, action):
        # Increment the step counter
        global current_step, max_steps
        current_step += 1

        # Perform the action and get the reward
        reward, terminated = self.game.move_hands_simultaneously(action[0], action[1])
        observation = self._get_observation()

        done = terminated  # This could be set based on the game's termination condition

        # Check if the episode should be truncated
        truncated = current_step >= max_steps
        if truncated:
            done = True  # Ensure "done" is also True when truncating

        # Reset the step counter if the episode is done
        if done:
            current_step = 0
            print(f'{reward}')



        info = {}
        return observation, reward, done, truncated, info



    def render(self):
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

    def seed(self, seed=None):
        # Optional: Set seed
        pass
