"""
1-step A2C (old release)
"""
import gym
import argparse
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedLayer(nn.Module):
    def __init__(self, input_dim, hidden):
        super(SharedLayer, self).__init__()
        self.fc1= nn.Linear(input_dim, hidden)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        return out


class Actor(nn.Module):
    def __init__(self, input_dim, shared_layer, out_dim):
        super(Actor, self).__init__()
        self.shared_layer = shared_layer
        self.fc2 = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        x = torch.tensor(x)
        x = self.shared_layer(x)
        out = F.log_softmax(self.fc2(x), dim=-1)
        return out


class Critic(nn.Module):
    def __init__(self, input_dim, shared_layer, out_dim=1):
        super(Critic, self).__init__()
        self.shared_layer = shared_layer
        self.fc2 = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        x = torch.tensor(x)
        x = self.shared_layer(x)
        out = self.fc2(x)
        return out


def choose_action(prob, action_dim):
    action = np.random.choice(a=action_dim, p=prob[0].detach().numpy())
    return action

def train_critic(critic_optim, critic, sample):
    obs, one_hot_action, log_probs, reward, next_obs = sample
    obs_v = critic(obs)
    next_obs_v = critic(next_obs)
    # (Q-V)^2
    critic_loss = (reward + args.gamma * next_obs_v.item() - obs_v)**2
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

def train_actor(actor_optim, critic, sample):
    obs, one_hot_action, log_probs, reward, next_obs = sample
    obs_v = critic(obs)
    next_obs_v = critic(next_obs)
    advantage = reward + next_obs_v.detach() - obs_v.detach()
    one_hot_action = torch.tensor(one_hot_action).unsqueeze(0)
    # -A*log(pi(a|s))
    actor_loss = -advantage * torch.sum(log_probs * one_hot_action, dim=-1)
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()


def main():
    # init env
    env = gym.make(args.env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # init network
    shared_layer = SharedLayer(obs_dim, args.hidden)
    actor = Actor(input_dim=args.hidden, shared_layer=shared_layer, out_dim=action_dim)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.a_lr)
    critic = Critic(input_dim=args.hidden, shared_layer=shared_layer, out_dim=1)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.c_lr)
    reward_list = []

    for i in range(args.episode):
        obs = env.reset()
        done = False
        step = 0
        episode_reward = 0
        while not done:
            step += 1
            # env.render()
            obs = np.expand_dims(obs, axis=0)
            log_probs = actor(obs)
            probs = torch.exp(log_probs)
            action = choose_action(probs, action_dim)
            one_hot_action = torch.eye(action_dim)[action].unsqueeze(dim=0)
            next_obs, reward, done, info = env.step(action)
            if done:
                reward = -10.0
            episode_reward += reward

            # training
            sample = obs, one_hot_action, log_probs, reward, next_obs
            train_critic(critic_optim, critic, sample)
            train_actor(actor_optim, critic, sample)
            obs = next_obs
        wandb.log({"reward": episode_reward})
        reward_list.append(episode_reward)
        if i % args.log_freq == 0:
            print("Episode:%d , reward: %f" % (i, episode_reward))
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id",      default="CartPole-v0", type=str)
    parser.add_argument("--a_lr",           default=1e-3,       type=float)
    parser.add_argument("--c_lr",           default=1e-3,       type=float)
    parser.add_argument("--hidden",         default=32,         type=int)
    parser.add_argument("--episode",        default=2000,       type=int)
    parser.add_argument("--gamma",          default=0.9,        type=float)
    parser.add_argument("--batch_size",     default=50,         type=int)
    parser.add_argument("--log_freq",       default=50,         type=int)
    parser.add_argument("--wandb_log",      default=False,      type=bool)
    args = parser.parse_args()
    if args.wandb_log:
        wandb.init(project="A2C", config=args)
        args = wandb.config
    main()
