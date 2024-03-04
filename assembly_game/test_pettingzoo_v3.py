from gym_env.ma_computer_assembly_env_v3 import env, ComputerAssemblyMultiAgentEnv
from pettingzoo.test import parallel_api_test

import os
from torch import nn
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()

def env_creator():
    env = ComputerAssemblyMultiAgentEnv()
    # env = MultiAgentEnvCompatibility(env)
    return env
if __name__ == "__main__":
    ray.init()

    # env = ComputerAssemblyMultiAgentEnv()
    # parallel_api_test(env, num_cycles=1_000_000)
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)



    env_name = "ComputerAssemblyEnv-v3"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator()))
    test_env = PettingZooEnv(env_creator())

    obs_space = test_env.observation_space
    act_space = test_env.action_space


    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .multi_agent(
            policies={
                "agent_1": (None, obs_space, act_space, {}),
                "agent_2": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )
    config.environment(disable_env_checking=True)
    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )
