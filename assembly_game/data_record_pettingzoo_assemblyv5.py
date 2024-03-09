import time

import pygame
import sys
from tianshou.env.pettingzoo_env import PettingZooEnv
from gym_env.ma_computer_assembly_env_AEC_v5 import ComputerAssemblyMultiAgentEnv
from tianshou.data import ReplayBuffer, Batch
def make_env():
    env = ComputerAssemblyMultiAgentEnv(render_mode="human")
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


# Initialize buffer
buffer_size = 10000  # Adjust this size according to your needs
buffer = ReplayBuffer(buffer_size)

def record_step(init_obs, current_action_agent1, current_action_agent2, obs, reward, terminated, truncated):
    buffer.add(Batch(obs=init_obs, act=[current_action_agent1, current_action_agent2],
                     rew=reward, terminated=terminated , truncated = truncated, obs_next=obs))

def save_buffer_to_hdf5(file_path):
    buffer.save_hdf5(file_path)



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
    time.sleep(0.2)
    env.render()



    # Environment interaction logic
    # # Simulate environment step with the current actions of both agents
    # action = {'agent_1':(current_action_agent1),
    #           'agent_2': (current_action_agent2)}
    observation, reward, term, truncated, info  = env.step(current_action_agent1+1)
    obs, reward, term, truncated, info = env.step(current_action_agent2+1)



    # Record the step
    record_step(init_obs, current_action_agent1+1, current_action_agent2+1, observation, reward, term,truncated)

    if term or truncated:
        observation = env.reset()  # Reset the environment for the next episode
        data_records = []
    current_action_agent2 = -1
    current_action_agent1 = -1
    init_obs = observation

save_buffer_to_hdf5('./data/recorded_data0228.hdf5')  # Save recorded data at the end of each episode

# Cleanup
pygame.quit()
env.close()
sys.exit()
