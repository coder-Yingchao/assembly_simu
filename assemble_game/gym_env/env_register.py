import gym
from gym.envs.registration import register

register(
    id='ComputerAssemblyEnv-v0',  # Use an environment ID with a version number
    entry_point='gym_env.computer_assembly_env:ComputerAssemblyEnv',  # Module path : ClassName
)
