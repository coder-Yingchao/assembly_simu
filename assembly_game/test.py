import gymnasium as gym
import gym_env.computer_assembly_env  # Ensure this matches the path to your Gym environment

# Create the Gym environment
env = gym.make('ComputerAssemblyEnv-v0')

# Reset the environment to start a new episode
observation = env.reset()

# Example actions to test moving hands simultaneously
# 9 represents moving to the case, and numbers 1-8 represent moving to components
actions = [(0, 9),(8,9),(1,10)]

for action in actions:
    observation, reward, done, truncated, info = env.step(action)

    # env.render()

# Close the environment
# import time

# After your loop
# time.sleep(5)  # Keeps the window open for 5 seconds


env.close()
