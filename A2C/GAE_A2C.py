"""
A2C using GAE
"""

import gym
import argparse
import wandb
import torch
import numpy as np
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
        x = Tensor(x).to(device)
        x = F.relu(self.fc(x))
        prob = F.softmax(self.actor(x), dim=-1)
        log_prob = torch.log(prob)
        return prob, log_prob

    def critic_forward(self, x):
        x = Tensor(x).to(device)
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
        self.net = Net(input_size, hidden_size, output_size).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.buffer = ReplayBuffer(capacity=args.capacity)

    def actor(self, s):
        prob, log_prob = self.net.actor_forward(s)
        return prob, log_prob

    def critic(self, s):
        return self.net.critic_forward(s)

    def push(self, s, a, log_pi, r, s_, done):
        self.buffer.push(s, a, log_pi, r, s_, done)

    def delta(self, s, s_, r, done):
        obs_v = agent.critic(s)
        next_obs_v = agent.critic(s_)
        return r + args.gamma * next_obs_v * (1 - done) - obs_v

    def train(self):
        sample = self.buffer.memory
        batch = Transition(*zip(*sample))

        obs_batch = Tensor(np.array(batch.obs))  # (5,4)
        log_pi_batch = torch.stack(batch.log_pi)  # (5)
        reward_batch = Tensor(batch.reward)  # (5)
        next_obs_batch = Tensor(np.array(batch.next_obs))  # (5,4)
        done_batch = Tensor(batch.done)  # (5)

        # gae bootstrapping
        obs_v = agent.critic(obs_batch)  # v(s)
        T = len(batch.reward)
        gae = torch.zeros_like(reward_batch).to(device)
        future_gae = torch.tensor(0.0)
        coef = args.gamma * args.lam
        for t in reversed(range(T)):
            future_gae = self.delta(obs_batch[t], next_obs_batch[t], reward_batch[t], done_batch[t]) +\
                  coef * (1-done_batch[t]) * future_gae
            gae[t] = future_gae
        advantage = gae.detach()
        v_target = advantage + obs_v
        loss = -advantage * log_pi_batch + (v_target.detach() - obs_v)**2
        self.optim.zero_grad()
        loss.mean().backward()
        self.optim.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",             default=1e-3,           type=float)
    parser.add_argument("--hidden",         default=32,             type=int)
    parser.add_argument("--num_episode",    default=10000,          type=int)
    parser.add_argument("--gamma",          default=0.9,            type=float)
    parser.add_argument("--batch_size",     default=32,             type=int)
    parser.add_argument("--log_freq",       default=20,             type=int)
    parser.add_argument("--n_step",         default=5,              type=int)
    parser.add_argument("--capacity",       default=10000,          type=int)
    parser.add_argument("--lam",            default=0.8,            type=float)
    parser.add_argument("--train_freq",     default=5,              type=int)
    parser.add_argument("--wandb_log",      action='store_true')
    args = parser.parse_args()
    if args.wandb_log:
        wandb.init(project="A2C_Lunar", config=args, name="GAE")
        args = wandb.config
    Transition = namedtuple(
        "Transition",
        (
            "obs", "action", "log_pi", "reward", "next_obs", "done"
        )
    )
    # init env
    env = gym.make("CartPole-v0")
    device = 'cpu'  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = A2C(obs_dim, args.hidden, action_dim)
    
    for i_episode in range(args.num_episode):
        obs, info = env.reset()
        done = False
        step = 0
        episode_reward = 0
        while not done:
            env.render()
            step += 1
            prob, log_prob = agent.actor(obs)
            action = int(prob.multinomial(1))
            next_obs, reward, terminated, truncated , info = env.step(action)
            done = terminated or truncated
            log_pi = log_prob[action]
            if done:
                reward = -20.0
            agent.push(obs, action, log_pi, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward
        agent.train()
        agent.buffer.clean()
        if args.wandb_log:
            wandb.log({"reward": episode_reward}, step=i_episode)
        if i_episode % args.log_freq == 0:
            print("Episode: %d, Reward: %f" % (i_episode, episode_reward))
