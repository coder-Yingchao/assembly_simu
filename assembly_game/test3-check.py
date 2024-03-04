from stable_baselines3.common.env_checker import check_env
import gym_env.computer_assembly_env  # Ensure this matches the path to your Gym environment
import gymnasium as gym

env = gym.make('ComputerAssemblyEnv-v0')
# It will check your custom environment and output additional warnings if needed
check_env(env)