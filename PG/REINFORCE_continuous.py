import gym
import wandb
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Normal
from collections import namedtuple


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = Tensor(x)
        x = F.relu(self.fc1(x))
        prob = torch.tanh(self.fc2(x))
        log_prob = torch.log(prob)
        return prob, log_prob


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def clean(self):
        self.position = 0
        self.memory.clear()

        
class PG:
    def __init__(self, input_size, hidden_size, output_size):
        self.net = Net(input_size, hidden_size, output_size)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.buffer = ReplayBuffer(capacity=args.capacity)

    def act(self, s):
        prob, log_prob = self.net(s)
        return prob, log_prob

    def push(self, s, a, log_pi, r, s_, done):
        self.buffer.push(s, a, log_pi, r, s_, done)

    def train(self):
        sample = self.buffer.memory
        batch = Transition(*zip(*sample))
        obs_batch = Tensor(batch.obs)  # (5,4)
        log_pi_batch = torch.stack(batch.log_pi)  # (5)
        reward_batch = Tensor(batch.reward)  # (5)
        next_obs_batch = Tensor(batch.next_obs)  # (5,4)
        done_batch = Tensor(batch.done)  # (5)

        # Monte-Carlo bootstrapping
        T = len(batch.reward)
        rets = torch.zeros_like(reward_batch)
        future_ret = 0.0
        for t in reversed(range(T)):
            future_ret = reward_batch[t] + args.gamma * future_ret
            rets[t] = future_ret

        loss = -rets * log_pi_batch
        self.optim.zero_grad()
        loss.mean().backward()
        self.optim.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",             default=1e-3,       type=float)
    parser.add_argument("--hidden",         default=32,         type=int)
    parser.add_argument("--num_episode",    default=10000,      type=int)
    parser.add_argument("--gamma",          default=0.9,        type=float)
    parser.add_argument("--batch_size",     default=32,         type=int)
    parser.add_argument("--log_freq",       default=20,         type=int)
    parser.add_argument("--n_step",         default=10,         type=int)
    parser.add_argument("--capacity",       default=10000,      type=int)
    config = parser.parse_args()
    wandb.init(project="PG", config=config, name="REINFORCE_continuous")
    args = wandb.config
    Transition = namedtuple(
        "Transition",
        (
            "obs", "action", "log_pi", "reward", "next_obs", "done"
        )
    )

    # init env
    env = gym.make("LunarLander-v2")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PG(obs_dim, args.hidden, action_dim)
    global_step = 0

    for i_episode in range(args.num_episode):
        obs = env.reset()
        done = False
        step = 0
        episode_reward = 0
        while not done:
            global_step += 1
            step += 1
            prob, log_prob = agent.act(obs)
            # dist = Categorical(prob)
            dist = Normal(loc=prob[0], scale=prob[1])
            action = dist.sample()
            log_pi = dist.log_prob(action)
            action = [action.item()]
            next_obs, reward, done, info = env.step(action)
            if done:
                reward = -20.0
            agent.push(obs, action, log_pi, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward

        agent.train()
        agent.buffer.clean()
        wandb.log({"reward": episode_reward}, step=i_episode)
        if i_episode % args.log_freq == 0:
            print("Episode: %d, Reward: %f" % (i_episode, episode_reward))
