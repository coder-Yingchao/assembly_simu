import gymnasium.utils.env_checker
from ray import tune, air

from ray.rllib.algorithms.ppo import PPOConfig
from gym_env.ma_computer_assembly_env_v2 import ComputerAssemblyMultiAgentEnv # Ensure this matches the path to your Gym environment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-gpus",
    type=int,
    default=1,
    help="Number of GPUs to use for training.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: Only one episode will be "
         "sampled.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.as_test:
        # Only a compilation test of running waterworld / independent learning.
        stop = {"training_iteration": 1}
    else:
        stop = {"episodes_total": 60000}

    env = ComputerAssemblyMultiAgentEnv()

    # gymnasium.utils.env_checker.check_env(env)
    config = (
        PPOConfig()
        .environment("computer_assembly_multi_agent_env")
        .resources(num_gpus=args.num_gpus)
        .rollouts(num_rollout_workers=0)
        .multi_agent(
            policies={
                "agent_1": (None, env.observation_space, env.action_space, {}),
                "agent_2": (None, env.observation_space, env.action_space, {}),
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
    )
    # config.environment(disable_env_checking=True)

    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
            ),
        ),
        param_space=config,
    ).fit()


