import argparse
import gym
import torch
import wandb
import numpy as np
import torch.nn.functional as F
from torch import nn, optim, Tensor
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = Tensor(x)
        x = F.relu(self.fc1(x))
        prob = F.softmax(self.fc2(x), dim=-1)
        return prob


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = Tensor(x)
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, *transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self):
        return zip(*self.buffer)

    def clean(self):
        return self.buffer.clear()


class PPO:
    def __init__(self, n_state, n_action):
        self.n_state = n_state
        self.n_action = n_action
        self._build()

    def _build(self):
        self.buffer = ReplayBuffer(args.capacity)
        self.actor = Actor(self.n_state, args.hidden, self.n_action)
        self.critic = Critic(self.n_state, args.hidden)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.a_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.c_lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        prob = self.actor(state)
        dist = Categorical(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action), log_prob

    def store_transition(self, *transition):
        self.buffer.push(*transition)

    def train(self):
        s, a, r, s_, done, old_log_prob = self.buffer.sample()
        s = torch.Tensor(np.stack(s))
        r = torch.Tensor(r).view(-1, 1)
        a = torch.Tensor(a)
        done = torch.IntTensor(done).view(-1, 1)
        old_log_prob = torch.stack(old_log_prob).detach()

        # cal target using old critic
        with torch.no_grad():
            s_ = np.array(s_)
            target = r + args.gamma * self.critic(s_) * (1 - done)
            delta = target - self.critic(s)
            delta = delta.numpy()
            adv_list = []
            adv = 0.0
            for delta_t in delta[::-1]:
                adv = args.gamma * args.lam * adv + delta_t
                adv_list.append(adv)
            adv_list.reverse()
            adv_list = Tensor(adv_list)

        for _ in range(args.n_update):
            self.update_actor(s, a, adv, old_log_prob)
            self.update_critic(s, target)
            
    def update_actor(self, state, action, adv, old_log_prob):
        # cal actor loss
        prob = self.actor(state)
        dist = Categorical(prob)
        log_prob = dist.log_prob(action)
        ratio = torch.exp(log_prob - old_log_prob)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1-args.epsilon, 1+args.epsilon) * adv
        actor_loss = -torch.mean(torch.min(surr1, surr2))
        if args.wandb_log:
            wandb.log({"Actor Loss": actor_loss})
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
    def update_critic(self, state, target):
        v = self.critic(state)
        critic_loss = self.loss_fn(v, target)
        if args.wandb_log:
            wandb.log({"Critic Loss": critic_loss})
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        

def main():
    env = gym.make("CartPole-v0")
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    agent = PPO(n_state, n_action)

    for i_episode in range(args.n_episodes):
        s, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            a, prob = agent.choose_action(s)
            s_, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            agent.store_transition(s, a, r, s_, done, prob)
            s = s_
            total_reward += r
        if args.wandb_log:
            wandb.log({"Reward": total_reward})
        if i_episode % 10 == 0:
            print(f"Episode: {i_episode}, Reward: {total_reward}")

        agent.train()
        agent.buffer.clean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", default=3000,       type=int)
    parser.add_argument("--a_lr",       default=1e-3,       type=float)
    parser.add_argument("--c_lr",       default=1e-3,       type=float)
    parser.add_argument("--hidden",     default=64,         type=int)
    parser.add_argument("--gamma",      default=0.99,       type=float)
    parser.add_argument("--epsilon",    default=0.2,        type=float)
    parser.add_argument("--capacity",   default=10000,      type=int)
    parser.add_argument("--n_update",   default=10,         type=int)
    parser.add_argument("--lam",        default=0.8,        type=float)
    parser.add_argument("--wandb_log",  action="store_true")
    args = parser.parse_args()
    if args.wandb_log:
        wandb.init(project="PPO_CartPole", config=args, name="PPO_GAE")
        args = wandb.config
    main()
