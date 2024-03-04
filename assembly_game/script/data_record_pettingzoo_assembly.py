import time

import pygame
import gymnasium as gym
import csv
import sys
import os
import os
import argparse
from tianshou.utils import TensorboardLogger

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import DQNPolicy, MultiAgentPolicyManager
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net

from gym_env.ma_computer_assembly_env_v3 import ComputerAssemblyMultiAgentEnv
from pettingzoo.utils import parallel_to_aec
from torch.utils.tensorboard import SummaryWriter



def make_env():
    env = ComputerAssemblyMultiAgentEnv(render_mode="human")
    env = parallel_to_aec(env)
    env = PettingZooEnv(env)
    return env

pygame.init()
env = make_env()
observation,info = env.reset()

screen = pygame.display.set_mode((1500, 1000))
pygame.display.set_caption('Gym Environment Interaction')

# Define action mappings for both agents
agent1_actions = {
    pygame.K_0: 0, pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3,
    pygame.K_4: 4, pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7,
    pygame.K_8: 8, pygame.K_9: 9
}
agent2_actions = {
    pygame.K_KP0: 0, pygame.K_KP1: 1, pygame.K_KP2: 2, pygame.K_KP3: 3,
    pygame.K_KP4: 4, pygame.K_KP5: 5, pygame.K_KP6: 6, pygame.K_KP7: 7,
    pygame.K_KP8: 8, pygame.K_KP9: 9
}

# Initialize current actions with a neutral value within your action space if necessary
current_action_agent1 = -1
current_action_agent2 = -1

data_records = []

def record_step(init_obs, current_action_agent1, current_action_agent2, obs, reward, done):
    data_records.append({
        'init_obs' : init_obs,
        'action_agent1': current_action_agent1,
        'action_agent2': current_action_agent2,
        'observation': obs,
        'reward': reward,
        'done': done
    })




def save_data_to_csv(file_path):
    # Check if the file exists and has content
    file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0

    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['init_obs', 'action_agent1', 'action_agent2', 'observation', 'reward',
                                                  'done'])

        # Write the header only if the file does not exist or is empty
        if not file_exists:
            writer.writeheader()

        for record in data_records:
            writer.writerow(record)


# Game loop
running = True
init_obs  = observation
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in agent1_actions:
                current_action_agent1 = agent1_actions[event.key]
            elif event.key in agent2_actions:
                current_action_agent2 = agent2_actions[event.key]
    time.sleep(1)
    env.render()



    # Environment interaction logic
    # # Simulate environment step with the current actions of both agents
    # action = {'agent_1':(current_action_agent1),
    #           'agent_2': (current_action_agent2)}
    env.step(current_action_agent1+1)
    observation, reward, done, truncated, info = env.step(current_action_agent2+1)


    # Record the step
    record_step(init_obs, current_action_agent1, current_action_agent2, observation, reward, done)

    if done:
        observation = env.reset()  # Reset the environment for the next episode
        save_data_to_csv('../data/recorded_data0228.csv')  # Save recorded data at the end of each episode
        data_records = []
    current_action_agent2 = -1
    current_action_agent1 = -1
    init_obs = observation

# Cleanup
pygame.quit()
env.close()
# save_data_to_csv('recorded_data.csv')  # Final save when exiting the game loop
sys.exit()
