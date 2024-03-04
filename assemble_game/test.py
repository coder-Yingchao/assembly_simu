import gym
import gym_env.computer_assembly_env  # Ensure this matches the path to your Gym environment

# Create the Gym environment
env = gym.make('ComputerAssemblyEnv')

# Reset the environment to start a new episode
observation = env.reset()

# Example actions to test moving hands simultaneously
# 9 represents moving to the case, and numbers 1-8 represent moving to components
actions = [(1, 9), (2, 8), (3, 7)]  # Example actions: Hand1 to component 1 and Hand2 to the case, etc.

for action in actions:
    observation, reward, done, info = env.step(action)
    env.render()

# Close the environment
env.close()
