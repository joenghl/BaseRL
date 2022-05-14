import gym
import wandb
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def len(self):
        return len(self.buffer)

    def push(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, n):
        index = np.random.choice(len(self.buffer), n)
        batch = [self.buffer[i] for i in index]
        return zip(*batch)

    def clean(self):
        self.buffer.clear()


class DQN:
    def __init__(self, env, input_size, hidden_size, output_size):
        self.env = env
        self.eval_net = QNet(input_size, hidden_size, output_size)
        self.target_net = QNet(input_size, hidden_size, output_size)
        self.optim = optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.eps = args.eps
        self.buffer = ReplayBuffer(args.capacity)
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0
    
    def choose_action(self, obs):
        if np.random.uniform() <= self.eps:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action_value = self.eval_net(obs)
            action = torch.max(action_value, dim=-1)[1].numpy()
        return int(action)

    def store_transition(self, *transition):
        self.buffer.push(*transition)
        
    def learn(self):
        if self.eps > args.eps_min:
            self.eps *= args.eps_decay

        if self.learn_step % args.update_target == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1 
        
        obs, actions, rewards, next_obs, dones = self.buffer.sample(args.batch_size)
        actions = torch.LongTensor(actions)  # LongTensor to use gather latter
        dones = torch.IntTensor(dones)
        rewards = torch.FloatTensor(rewards)

        q_eval = self.eval_net(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_next = self.target_net(next_obs).detach()
        q_target = rewards + args.gamma * (1 - dones) * torch.max(q_next, dim=-1)[0]  # Q_target = r + gamma * q_next
        loss = self.loss_fn(q_eval, q_target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


def main():
    env = gym.make(args.env)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    agent = DQN(env, o_dim, args.hidden, a_dim) 
    for i_episode in range(args.n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action) 
            agent.store_transition(obs, action, reward, next_obs, done)
            episode_reward += reward
            obs = next_obs
            if agent.buffer.len() >= args.capacity:
                agent.learn()
        if args.wandb_log:
            wandb.log({"Reward": episode_reward}, step=i_episode)
        print(f"Episode: {i_episode}, Reward: {episode_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        default="CartPole-v0",  type=str)
    parser.add_argument("--lr",             default=1e-3,       type=float)
    parser.add_argument("--hidden",         default=64,         type=int)
    parser.add_argument("--n_episodes",     default=2000,       type=int)
    parser.add_argument("--gamma",          default=0.99,       type=float)
    parser.add_argument("--log_freq",       default=100,        type=int)
    parser.add_argument("--capacity",       default=10000,      type=int)
    parser.add_argument("--eps",            default=1.0,        type=float)
    parser.add_argument("--eps_min",        default=0.05,       type=float)
    parser.add_argument("--batch_size",     default=128,        type=int)
    parser.add_argument("--eps_decay",      default=0.999,      type=float)
    parser.add_argument("--update_target",  default=100,        type=int)
    parser.add_argument("--wandb_log",      default=False,      type=bool)
    args = parser.parse_args()
    if args.wandb_log:
        wandb.init(project="DQN_CartPole", config=args, name="DQN")
        args = wandb.config
    main()
