import gymnasium as gym
from tianshou.env.pettingzoo_env import PettingZooEnv

from gym_env.ma_computer_assembly_env_AEC_v5 import ComputerAssemblyMultiAgentEnv

env = ComputerAssemblyMultiAgentEnv(render_mode="")
env = PettingZooEnv(env)
# Reset the environment to start a new episode
observation = env.reset()

# Example actions to test moving hands simultaneously
# 9 represents moving to the case, and numbers 1-8 represent moving to components
# actions = [(0, 9),(8,9),(1,10)]
actions = [10,1, 10 ,1]
for action in actions:
    observation, reward, done, truncated, info = env.step(action)

    # env.render()

# Close the environment
# import time

# After your loop
# time.sleep(5)  # Keeps the window open for 5 seconds


env.close()
