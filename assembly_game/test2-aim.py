import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import gym_env.computer_assembly_env  # Ensure this matches the path to your Gym environment
from aim.sb3 import AimCallback
# Instantiate the environment
env_id = "ComputerAssemblyEnv-v0"
env = gym.make(env_id)
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=500000, callback=AimCallback(experiment_name='example2'))

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
