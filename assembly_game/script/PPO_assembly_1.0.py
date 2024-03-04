
from tianshou.utils import TensorboardLogger
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import PPOPolicy, MultiAgentPolicyManager
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils.net.common import ActorCritic
from gym_env.ma_computer_assembly_env_v3 import ComputerAssemblyMultiAgentEnv
from pettingzoo.utils import parallel_to_aec
import numpy as np
import torch
from tianshou.utils.net.common import Net




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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden_size", type=int, nargs="*", default=[128, 128, 128, 128]
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
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def make_env():
    env = ComputerAssemblyMultiAgentEnv(render_mode="")
    env = parallel_to_aec(env)
    env = PettingZooEnv(env)
    return env

if __name__ == "__main__":
    # Load and wrap the environment
    env = make_env()
    args: argpartrain_collectorse.Namespace = get_args()

    # Convert the env to vector format and create a collector
    envs = DummyVectorEnv([lambda: make_env() for _ in range(1)])
    test_envs = DummyVectorEnv([lambda: make_env() for _ in range(1)])

    # Observation and action space

    observation_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n

    pre_net = Net(observation_shape, 512,  device="cuda" if torch.cuda.is_available() else "cpu").to("cuda" if torch.cuda.is_available() else "cpu")


    actor =  Actor(preprocess_net=pre_net, action_shape=action_shape, hidden_sizes=args.hidden_size, softmax_output= False, preprocess_net_output_dim=512,device="cuda" if torch.cuda.is_available() else "cpu")
    critic = Critic(preprocess_net=pre_net,hidden_sizes=args.hidden_size, last_size=1,device="cuda" if torch.cuda.is_available() else "cpu")

    # Combine actor and critic parameters for the optimizer
    optimizer1 = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=0.001,  eps=1e-5)


    # define policy


    policy1 = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optimizer1,
        dist_fn=torch.distributions.Categorical,
        eps_clip=0.2,
        discount_factor= 0.9,
        action_space = env.action_space,
        action_scaling = False
        # Other parameters as needed
    )

    actor2 = Actor(preprocess_net=pre_net, action_shape=action_shape, hidden_sizes=args.hidden_size,
                  softmax_output=False, preprocess_net_output_dim=512,
                  device="cuda" if torch.cuda.is_available() else "cpu")
    critic2 = Critic(preprocess_net=pre_net, hidden_sizes=args.hidden_size, last_size=1,
                    device="cuda" if torch.cuda.is_available() else "cpu")

    # critic2 = CriticNet(observation_shape)

    # Combine actor and critic parameters for the optimizer
    optimizer2 = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=0.001,  eps=1e-5)
    policy2 = PPOPolicy(
        actor=actor2,
        critic=critic2,
        optim=optimizer2,
        dist_fn=torch.distributions.Categorical,
        eps_clip=0.2,
        action_space=env.action_space,
        action_scaling=False
    )
    policy =[policy1, policy2]
    policies = MultiAgentPolicyManager(
        policies=policy,
        env=env)
    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policies,
        envs,
        VectorReplayBuffer(20_000, len(envs)),
        exploration_noise=True,
    )
    train_collector.collect(n_step=64 * 10)  # batch size * training_num

    test_collector = Collector(policies, test_envs, exploration_noise=True)
    # test_collector.collect(n_step=640)
    # ======== tensorboard logging setup =========
    log_path = os.path.join(args.logdir, "assemblygame", "ppo_1.0")
    writer = SummaryWriter(log_path)
    # Assuming `acc_reward` is your accumulated reward for the episode and `global_step is a step counter
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # Training
    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path = os.path.join("log", "assembly", "dqn", "policy.pth")
        os.makedirs(os.path.join("log", "assembly", "dqn"), exist_ok=True)
        torch.save(policy.policies[env.agents[1]].state_dict(), model_save_path)


    def log_episode_rewards(collector, phase, global_step, writer):
        episode_rewards = collector.buffer.rew
        if episode_rewards is not None and len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards)
            writer.add_scalar(f"{phase}/average_episode_reward", avg_reward, global_step)

    def stop_fn(mean_rewards):
        return mean_rewards >= 500

    def train_fn(epoch, env_step):
        eps = max(0.1, 1 - epoch * 0.00003)  # Example of linearly decreasing epsilon

        policies.policies[env.agents[0]].set_eps(eps)
        policies.policies[env.agents[1]].set_eps(eps)
        log_episode_rewards(train_collector, "train", epoch, writer)



    def test_fn(epoch, env_step):
        policies.policies[env.agents[0]].set_eps(0.05)
        policies.policies[env.agents[1]].set_eps(0.05)
        log_episode_rewards(test_collector, "test", epoch, writer)


    def reward_metric(rews):
        return rews[:, 1]
    # ======== Step 5: Run the trainer =========
    # For PPO, we use onpolicy_trainer instead of offpolicy_trainer
    result = OnpolicyTrainer(
        policy=policies,
        train_collector=train_collector,
        test_collector=None,
        max_epoch=500,

        step_per_epoch=1000,
        episode_per_test=10,
        batch_size = 64,
        # train_fn=train_fn,
        # test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
        repeat_per_collect = 2
    )


    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")