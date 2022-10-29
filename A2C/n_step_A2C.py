"""
n-steps A2C algorithms
"""

import gym
import argparse
import numpy as np
import wandb
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.actor = nn.Linear(hidden_size, output_size)
        self.critic = nn.Linear(hidden_size, 1)

    def actor_forward(self, x):
        x = Tensor(x)
        x = F.relu(self.fc(x))
        prob = F.softmax(self.actor(x), dim=-1)
        log_prob = torch.log(prob)
        return prob, log_prob

    def critic_forward(self, x):
        x = Tensor(x)
        x = F.relu(self.fc(x))
        out = self.critic(x)
        return out


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


class A2C:
    def __init__(self, input_size, hidden_size, output_size):
        self.net = Net(input_size, hidden_size, output_size)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.buffer = ReplayBuffer(capacity=args.capacity)

    def actor(self, s):
        prob, log_prob = self.net.actor_forward(s)
        return prob, log_prob

    def critic(self, s):
        return self.net.critic_forward(s)

    def push(self, s, a, log_pi, r, s_, done):
        self.buffer.push(s, a, log_pi, r, s_, done)

    def train(self):
        sample = self.buffer.memory
        batch = Transition(*zip(*sample))
        obs_batch = Tensor(np.array(batch.obs))  # (5,4)
        action_batch = Tensor(batch.action)  # (5)
        log_pi_batch = batch.log_pi  # (5)
        reward_batch = Tensor(batch.reward)  # (5)
        next_obs_batch = Tensor(np.array(batch.next_obs))  # (5,4)
        done_batch = Tensor(batch.done)  # (5)

        # n-step bootstrapping
        obs_v = agent.critic(obs_batch[0])  # v(s)
        next_ret = agent.critic(next_obs_batch[-1])
        rets = torch.zeros_like(reward_batch)
        for t in reversed(range(args.n_step)):
            next_ret = reward_batch[t] + args.gamma * next_ret.item() * (1 - done_batch[t])
        q = next_ret
        advantage = q - obs_v.detach()
        loss = -advantage*log_pi_batch[0] + (q-obs_v)**2
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",             default=1e-3,           type=float)
    parser.add_argument("--hidden",         default=32,             type=int)
    parser.add_argument("--num_episode",    default=10000,          type=int)
    parser.add_argument("--gamma",          default=0.9,            type=float)
    parser.add_argument("--batch_size",     default=32,             type=int)
    parser.add_argument("--log_freq",       default=20,             type=int)
    parser.add_argument("--n_step",         default=10,             type=int)
    parser.add_argument("--capacity",       default=10000,          type=int)
    parser.add_argument("--wandb_log",      action="store_true")
    args = parser.parse_args()
    if args.wandb_log:
        wandb.init(project="A2C", config=args, name="n=10")
        args = wandb.config
    Transition = namedtuple(
        "Transition",
        (
            "obs", "action", "log_pi", "reward", "next_obs", "done"
        )
    )
     # init env
    env = gym.make("CartPole-v0")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = A2C(obs_dim, args.hidden, action_dim)
    global_step = 0
    
    for i_episode in range(args.num_episode):
        obs, info = env.reset()
        done = False
        step = 0
        episode_reward = 0
        while not done:
            global_step += 1
            step += 1
            prob, log_prob = agent.actor(obs)
            action = int(prob.multinomial(1))
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            log_pi = log_prob[action]
            if done:
                reward = -20.0
            agent.push(obs, action, log_pi, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward
            if step == args.n_step:
                step = 0
                agent.train()
                agent.buffer.clean()
        if global_step % args.n_step == 0 and args.wandb_log:
            wandb.log({"reward": episode_reward}, step=i_episode)
        if i_episode % args.log_freq == 0:
            print("Episode: %d, Reward: %f" % (i_episode, episode_reward))
