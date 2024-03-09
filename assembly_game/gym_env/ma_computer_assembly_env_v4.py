import numpy as np
from pettingzoo import ParallelEnv

from gymnasium import  spaces
from game.computer_assembly_game_v1 import ComputerAssemblyGame
from pettingzoo.utils import parallel_to_aec, wrappers
import gymnasium as gym

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # # This wrapper is only for environments which print results to the terminal
    # if render_mode == "ansi":
    #     env = wrappers.CaptureStdoutWrapper(env)
    # # this wrapper helps error handling for discrete action spaces
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    # env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env

    """

    env = ComputerAssemblyMultiAgentEnv(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class ComputerAssemblyMultiAgentEnv(ParallelEnv):
    metadata = {
        "name": "computer_assembly_multi_agent_env_pettingzoo",
    }

    def __init__(self,render_mode=None):
        super().__init__()  # Ensure proper initialization of base classes
        self.game = ComputerAssemblyGame()
        self.agents = ['agent_1', 'agent_2']
        self.possible_agents = self.agents[:]
        self.action_spaces = {agent: spaces.Discrete(11) for agent in self.agents}
        # Define observation space similarly, customized to your environment's needs
        n_components = 8
        obs_space_low = [0, 0, 1] * n_components + [0, 0, -1] * 2 + [0]
        obs_space_high = [self.game.screen_width / 100, self.game.screen_height / 100, 4] * n_components + [
            self.game.screen_width / 100, self.game.screen_height / 100, 9] * 2 + [1]
        # self.observation_spaces = {
        #     agent: spaces.Box(low=np.array(obs_space_low), high=np.array(obs_space_high), dtype=np.int32) for agent
        #     in self.agents}
        # Assuming screen dimensions and other components are discretized into fixed number of discrete steps
        discrete_screen_width_steps = 15  # Example: screen width divided into 8 discrete steps
        discrete_screen_height_steps = 10  # Example: screen height divided into 6 discrete steps
        other_component_steps = [4, 10, 1]  # Example: other observations with fixed discrete steps

        # Combine all steps into a single array for MultiDiscrete
        # multi_discrete_obs_space = [discrete_screen_width_steps,
        #                             discrete_screen_height_steps,4] * n_components + [discrete_screen_width_steps,
        #                             discrete_screen_height_steps,10]*2 +[1]

        multi_discrete_obs_space = [4] * n_components + [10]*2 +[1]

        self.observation_spaces = {
            agent: spaces.MultiDiscrete(np.array(multi_discrete_obs_space) ) for agent in self.agents
        }
        self.current_step = 0
        self.max_steps = 3000
        self.render_mode = render_mode
        self.last_step_return =[]



    def reset(self, seed=None, options=None):
        self.current_step = 0
        # Reset the game to its initial state
        # self.game.reset_game(seed)  # Assuming reset_game doesn't need a seed. If it does, pass seed here.
        observation = self._get_observation()
        observations = {
            a: (observation)
            for a in self.agents
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}
        # In a multi-agent environment, return observations for each agent
        return observations, infos

    def step(self, actions):
        # print(f'action:{actions}')
        self.current_step +=1

        action_agent1 = actions[self.agents[0]]-1
        action_agent2 = actions[self.agents[1]]-1

        reward, terminated = self.game.move_hands_simultaneously(action_agent1, action_agent2)


        observation = self._get_observation()
        observations = {
            a: (observation)
            for a in self.agents
        }
        rewards = {
            a: (reward)
            for a in self.agents
        }
        terminations = {
            a: (terminated)
            for a in self.agents
        }

        infos = {a: {} for a in self.agents}
        truncated = self.current_step >= self.max_steps
        truncations = {
            a: (truncated)
            for a in self.agents
        }
        self.last_step_return = observations, rewards, terminations, truncations, infos
        # print(f'step: {self.current_step},action_agent1, action_agent2:{action_agent1},{action_agent2} reward: {reward}')




        return observations, rewards, terminations, truncations, infos

    # def last(self, *args, **kwargs):
    #
    #     # Check if last_step_return has been set and has the correct format
    #     if hasattr(self, 'last_step_return') and len(self.last_step_return) == 5:
    #         return self.last_step_return
    #     else:
    #         # Fallback values in case last_step_return is not set or incorrect
    #         empty_observation = {}  # Adapt this based on your observation space
    #         empty_rewards = {agent: 0 for agent in self.agents}
    #         empty_terminations = {agent: False for agent in self.agents}
    #         empty_truncations = {agent: False for agent in self.agents}
    #         empty_infos = {agent: {} for agent in self.agents}
    #         return (empty_observation, empty_rewards, empty_terminations, empty_truncations, empty_infos)

    def render(self):
        if self.render_mode== 'human':
            self.game.render()


    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    def _get_observation(self):
        # Initialize an empty list to hold observation data
        obs = []

        # Loop through each component and append its position and state to the observation list
        for component_name, component_data in self.game.states.items():
            # obs.append(int(component_data['position'][0])/100)  # x position, cast to int
            # obs.append(int(component_data['position'][1])/100)  # y position, cast to int
            obs.append(int(component_data['state']))  # state, cast to int

        # obs.append(self.game.hand_x/100)
        # obs.append(self.game.hand_y/100)
        if self.game.action_hand1 is None:
            obs.append(-1)
        else:
            obs.append(self.game.action_hand1)
        # obs.append(self.game.second_hand_x/100)
        # obs.append(self.game.second_hand_y/100)
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



