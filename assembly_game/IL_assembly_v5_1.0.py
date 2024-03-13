import os
import argparse
from tianshou.utils import TensorboardLogger

import gymnasium
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import DQNPolicy, MultiAgentPolicyManager, ImitationPolicy
from tianshou.trainer import OfflineTrainer
from tianshou.utils.net.common import Net

from gym_env.ma_computer_assembly_env_AEC_v5 import ComputerAssemblyMultiAgentEnv
from torch.utils.tensorboard import SummaryWriter



# Define the network architecture
def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gamma", type=float, default=0.9, help="a smaller gamma favors earlier win"
    )
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--update-per-epoch", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
    )
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.1)
    parser.add_argument(
        "--win-rate",
        type=float,
        default=0.6,
        help="the expected winning rate: Optimal policy can get 0.7",
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, " "watch the play of pre-trained models",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=2,
        help="the learned agent plays as the"
        " agent_id-th player. Choices are 1 and 2.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="the path of agent pth file " "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--opponent-path",
        type=str,
        default="",
        help="the path of opponent agent pth file "
        "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--device", type=str, default="cpu"
    )
    parser.add_argument(
        "--load-buffer-name",
        type=str,
        default="./data/recorded_data0310_2.hdf5",
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]




def make_env():
    env = ComputerAssemblyMultiAgentEnv(render_mode="")
    env = PettingZooEnv(env)
    return env

if __name__ == "__main__":
    # Load and wrap the environment
    env = make_env()
    args: argparse.Namespace = get_args()

    # api_test(env, num_cycles=1000, verbose_progress=False)

    # Convert the env to vector format and create a collector
    envs = DummyVectorEnv([lambda: make_env() for _ in range(1)])
    test_envs = DummyVectorEnv([lambda: make_env() for _ in range(1)])

    # Observation and action space

    observation_shape = env.observation_space.shape or env.observation_space.n,
    action_shape = env.action_space.n

    # Neural networks for each agent
    # Neural networks for each agent
    # net1 = DQNNet(observation_shape, action_shape)
    # net2 = DQNNet(observation_shape, action_shape)

    net1 = Net(
        state_shape= env.observation_space.shape or env.observation_space.n,
        action_shape=env.action_space.shape or env.action_space.n,
        hidden_sizes=[128, 128, 128, 128],
        device= args.device,
    ).to(args.device)

    net2 = Net(
        state_shape=env.observation_space.shape or env.observation_space.n,
        action_shape=env.action_space.shape or env.action_space.n,
        hidden_sizes=[128, 128, 128, 128],
        device= args.device,
    ).to(args.device)



    # Optimizers
    optim1 = torch.optim.Adam(net1.parameters(), lr=1e-3)
    optim2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    policy1: ImitationPolicy = ImitationPolicy(actor=net1, optim=optim1, action_space=env.action_space)
    policy2: ImitationPolicy = ImitationPolicy(actor=net2, optim=optim2, action_space=env.action_space)

    agents = [policy1,policy2]

    # Policy manager
    policies = MultiAgentPolicyManager(
        policies=agents,
        env= env)
    assert os.path.exists(
        args.load_buffer_name,
    ), "Please run data_record.py first to get expert's data buffer."


    buffer = ReplayBuffer.load_hdf5(args.load_buffer_name, args.device)



    print("Replay buffer size:", len(buffer))

    # ======== Step 3: Collector setup =========
    test_collector = Collector(policies, test_envs, exploration_noise=True)
    # ======== tensorboard logging setup =========
    log_path = os.path.join(args.logdir, "assemblygame_v5", "IL")
    writer = SummaryWriter(log_path)
    # Assuming `acc_reward` is your accumulated reward for the episode and `global_step is a step counter
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # Training
    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model1_save_path = os.path.join("log", "assembly", "IL", "policy.pth")
        model2_save_path = os.path.join("log", "assembly", "IL", "policy.pth")
        os.makedirs(os.path.join("log", "assembly", "IL"), exist_ok=True)
        torch.save(policy.policies[env.agents[0]].state_dict(), model1_save_path)
        torch.save(policy.policies[env.agents[1]].state_dict(), model2_save_path)



    def log_episode_rewards(collector, phase, global_step, writer):
        episode_rewards = collector.buffer.rew
        if episode_rewards is not None and len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards)
            writer.add_scalar(f"{phase}/average_episode_reward", avg_reward, global_step)

    def stop_fn(mean_rewards):
        return mean_rewards >= 1500

    def reward_metric(rews):
        return rews[:, 1]
    # ======== Step 5: Run the trainer =========
    result = OfflineTrainer(
        policy=policies,
        buffer=buffer,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.update_per_epoch,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()
    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
