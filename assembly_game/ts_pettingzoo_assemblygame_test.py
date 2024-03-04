import os
from typing import Optional, Tuple

import gymnasium
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net

from gym_env.ma_computer_assembly_env_v3 import env, ComputerAssemblyMultiAgentEnv
from pettingzoo.utils import parallel_to_aec, wrappers




if __name__ == "__main__":
    # Step 1: Load the PettingZoo environment
    env = ComputerAssemblyMultiAgentEnv(render_mode="human")
    env = parallel_to_aec(env)
    # Step 2: Wrap the environment for Tianshou interfacing
    env = PettingZooEnv(env)

    # Step 3: Define policies for each agent
    policies = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy()], env)

    # Step 4: Convert the env to vector format
    env = DummyVectorEnv([lambda: env])

    # Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
    collector = Collector(policies, env)

    # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
    result = collector.collect(n_episode=1, render=0.1)
