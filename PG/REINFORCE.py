import gym
import argparse
import numpy as np
import torch
import wandb
import torch.nn.functional as F
from torch import nn, optim, Tensor
from torch.distributions import Categorical


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = Tensor(x)
        x = F.relu(self.fc1(x))
        prob = F.softmax(self.fc2(x), dim=-1)
        return prob


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self):
        return zip(*self.buffer)

    def clean(self):
        self.buffer.clear()


class PG:
    def __init__(self, input_size, hidden_size, output_size):
        self.net = Net(input_size, hidden_size, output_size)
        self.optim = optim.Adam(self.net.parameters(), lr=args.lr)
        self.buffer = ReplayBuffer(capacity=args.capacity)
    
    def act(self, obs):
        prob = self.net(obs)
        dist = Categorical(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action), log_prob

    def push(self, *transition):
        self.buffer.push(*transition)

    def train(self):
        log_probs, rewards = self.buffer.sample()
        log_probs = torch.stack(log_probs)
        T = len(rewards)
        rets = np.empty(T, dtype=np.float32)
        future_rets = 0.0
        for t in reversed(range(T)):
            future_rets = rewards[t] + args.gamma * future_rets
            rets[t] = future_rets
        rets = torch.tensor(rets)
        loss = -rets * log_probs
        self.optim.zero_grad()
        loss.sum().backward()
        self.optim.step()


def main():
    env = gym.make("CartPole-v0")
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    agent = PG(o_dim, args.hidden, a_dim)
    for i_episode in range(args.n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, log_prob = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            agent.push(log_prob, reward)
            obs = next_obs
            episode_reward += reward
        agent.train()
        agent.buffer.clean()
        if args.wandb_log:
            wandb.log({"Reward": episode_reward}, step=i_episode)
        if i_episode % args.log_freq == 0:
            print(f"Episode: {i_episode}, Reward: {episode_reward}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",             default=1e-3,       type=float)
    parser.add_argument("--hidden",         default=32,         type=int)
    parser.add_argument("--n_episodes",     default=3000,       type=int)
    parser.add_argument("--gamma",          default=0.99,       type=float)
    parser.add_argument("--log_freq",       default=20,         type=int)
    parser.add_argument("--capacity",       default=10000,      type=int)
    parser.add_argument("--wandb_log",      default=False,      type=bool)
    args = parser.parse_args()
    if args.wandb_log:
        wandb.init(project="PG", config=args, name="REINFORCE_CartPole")
        args = wandb.config
    main()
