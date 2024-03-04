import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
import  random
import json
class OfflineEnv(gym.Env):
    """
    A custom Gym environment for offline training from a pre-recorded dataset.
    Focused on Agent 2's actions.
    """

    def __init__(self, df):
        super(OfflineEnv, self).__init__()
        self.df = df
        self.current_step = 0

        # Assuming observation space and action space are known
        # Adjust these based on your actual environment
        # n_components = 8  # Or however many components you have
        # obs_space_low = [1] * n_components + [0]
        # obs_space_high = [4] * n_components + [1]
        # self.observation_space = spaces.Box(low=np.array(obs_space_low), high=np.array(obs_space_high), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(11)  # 0-9 for Agent 2

        n_components = 8
        obs_space_low = [0, 0, 1] * n_components + [0, 0, -1] * 2 + [0]
        obs_space_high = [15, 10, 4] * n_components + [
            15,10, 9] * 2 + [1]
        self.observation_space = spaces.Box(low=np.array(obs_space_low), high=np.array(obs_space_high), dtype=np.int32)


    # def reset(self, seed=None, options=None):
    #     if seed is not None:
    #         np.random.seed(seed)
    #         random.seed(seed)
    #     # Reset the game to its initial state
    #     observation = self._next_observation()  # Get the initial game state as observation
    #     info = {}
    #     return observation, info

    def reset(self, seed=None):
        # Reset for a new episode
        self.current_step = 0
        observation = self._next_observation()
        # Ensure an 'info' dict is returned, even if empty
        return observation, {}

    def _next_observation(self):
        # Assuming observation is stored in a suitable format
        # Convert observation to the expected format here
        obs_str = self.df.loc[self.current_step, 'observation']
        valid_json_str = obs_str.replace(' ', ',')
        # valid_json_str = valid_json_str.replace('[,','[')
        valid_json_str = valid_json_str.replace(',,', ',')
        valid_json_str = valid_json_str.replace('\n', '')

        # obs_list = json.loads(valid_json_str)
        fixed_json_str = valid_json_str.replace('[,', '[', 1)
        # print(fixed_json_str)
        try:
            data = json.loads(fixed_json_str)
        except json.JSONDecodeError:
            print("Failed to load JSON from:", fixed_json_str)
            raise  # Optionally re-raise the exception or handle it

        obs = np.array(data, dtype=np.int32)
        return obs

    def step(self, action):
        obs = None  # Placeholder value or initial observation
        reward = 0
        done = False
        info = {}
        if self.current_step >= len(self.df) - 1:
            # Option 1: Automatically reset the environment (simulate new episode)
            observation, info = self.reset()
            done = True  # Indicate the episode is done
            reward = 0  # Optionally set a reward for episode end if appropriate
            # Option 2: Simply signal that the episode is done without resetting here
            # done = True
            # observation = self._next_observation()  # This would need to handle end-of-data gracefully
        else:
            # Simulate stepping in the environment using the pre-recorded data
            if action == self.df.loc[self.current_step, 'action_agent2']:
                reward = self.df.loc[self.current_step, 'reward']
            else:
                reward = 0  # Assuming no reward if the wrong action is taken, adjust as needed

            self.current_step += 1
            done = self.current_step == len(self.df)
            obs = self._next_observation()

        return obs, reward, done, False, info







# Load dataset
df = pd.read_csv('./data/recorded_data0217.csv')
import pandas as pd

# Load your dataset

# Identifying action and no-action instances
action_instances = df[df['action_agent2'] != -1]  # All instances where an action is taken
no_action_instances = df[df['action_agent2'] == -1]  # All instances where no action is taken

# Balancing the dataset
# Option 1: Undersample no-action instances
undersampled_no_action_instances = no_action_instances.sample(len(action_instances))

# Option 2: Oversample action instances (uncomment to use)
# oversampled_action_instances = action_instances.sample(len(no_action_instances), replace=True)

# Combining back into a balanced dataset
balanced_df = pd.concat([action_instances, undersampled_no_action_instances])
# balanced_df = pd.concat([action_instances, oversampled_action_instances])

# Shuffling the dataset to mix action and no-action instances
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

# Proceed with balanced_df for your environment setup and training

# Initialize the environment
env = OfflineEnv(df)
# env = OfflineEnv(balanced_df)
check_env(env)  # Optional, to check if the environment follows Gym API

# Vectorized environments wrap your env into a vectorized wrapper
# vec_env = DummyVecEnv([lambda: env])

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log= './offlineRL')
model.learn(total_timesteps=3e6)

# Save the model
model.save("dqn_agent2_model")
