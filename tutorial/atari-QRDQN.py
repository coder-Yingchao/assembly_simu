import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Create the environment
env_id = "Taxi-v3"
env = make_vec_env(env_id, n_envs=1)

# Initialize the agent
model = DQN('MlpPolicy', env, verbose=1, learning_rate=1e-3, buffer_size=50000, learning_starts=1000, batch_size=32, tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=1000, exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.1)

# Train the agent
total_timesteps = 10000
model.learn(total_timesteps=total_timesteps)

# Save the model
model.save("dqn_taxi")

# Load the model
# model = DQN.load("dqn_taxi")

# Evaluation




import time

# Make sure to create a new environment for evaluation
eval_env = gym.make("Taxi-v3")

# Evaluate the agent
episode_rewards = []
n_eval_episodes = 10
eval_env = gym.make(env_id)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
#
# # Calculate and print the mean reward
# mean_reward = sum(episode_rewards) / n_eval_episodes
# print(f"Mean reward: {mean_reward}")
#
# # Assuming eval_env is a single Gym environment
# for episode in range(n_eval_episodes):
#     state = eval_env.reset()
#     done = False
#     episode_reward = 0
#     while not done:
#         eval_env.render()
#         # No need for obs[None, :], just use obs directly if it's already the correct shape
#         action, _ = model.predict(state, deterministic=True)  # Adjust observation shape
#         state, reward, done, info = eval_env.step(action)
#         episode_reward += reward
#         time.sleep(0.1)  # Adjust this based on your preference
#     print(f"Episode Reward: {episode_reward}")
#
env.close()
eval_env.close()
#
