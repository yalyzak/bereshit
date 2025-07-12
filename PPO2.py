import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        hidden = self.fc(x)
        return self.actor(hidden), self.critic(hidden)

# class PPOAgent:
#     def __init__(self, obs_dim, action_dim, clip=0.2, gamma=0.99, lambd=0.95, lr=2.5e-4):
#         self.model = ActorCritic(obs_dim, action_dim)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
#
#         self.clip = clip
#         self.gamma = gamma
#         self.lambd = lambd
#
#         self.memory = []
#
#     def get_action(self, obs):
#         obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
#         logits, value = self.model(obs_t)
#         probs = torch.softmax(logits, dim=-1)
#         dist = torch.distributions.Categorical(probs)
#         action = dist.sample()
#         return action.item(), dist.log_prob(action), value.item()
#
#     def store(self, transition):
#         self.memory.append(transition)
#
#     def compute_gae(self, rewards, values, dones):
#         advantages = []
#         gae = 0
#         values = list(values) + [0.0]
#
#         for i in reversed(range(len(rewards))):
#             delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
#             gae = delta + self.gamma * self.lambd * (1 - dones[i]) * gae
#             advantages.insert(0, gae)
#         return advantages
#
#     def update(self, epochs=4, batch_size=64):
#         obs, actions, logprobs, rewards, values, dones = zip(*self.memory)
#         advantages = self.compute_gae(rewards, values, dones)
#         returns = [a + v for a, v in zip(advantages, values)]
#
#         obs = torch.tensor(obs, dtype=torch.float32)
#         actions = torch.tensor(actions)
#         old_logprobs = torch.tensor(logprobs)
#         advantages = torch.tensor(advantages, dtype=torch.float32)
#         returns = torch.tensor(returns, dtype=torch.float32)
#
#         for _ in range(epochs):
#             for i in range(0, len(obs), batch_size):
#                 batch_slice = slice(i, i+batch_size)
#                 logits, value_preds = self.model(obs[batch_slice])
#                 probs = torch.softmax(logits, dim=-1)
#                 dist = torch.distributions.Categorical(probs)
#
#                 new_logprobs = dist.log_prob(actions[batch_slice])
#                 ratio = torch.exp(new_logprobs - old_logprobs[batch_slice])
#
#                 surr1 = ratio * advantages[batch_slice]
#                 surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages[batch_slice]
#
#                 policy_loss = -torch.min(surr1, surr2).mean()
#                 value_loss = (returns[batch_slice] - value_preds.squeeze()).pow(2).mean()
#
#                 self.optimizer.zero_grad()
#                 (policy_loss + 0.5 * value_loss).backward()
#                 self.optimizer.step()
#
#         self.memory.clear()




class Agent:
    def __init__(self, obs_dim, action_dim, name="Agent", max_steps=100,optimizer=None):
        self.model = ActorCritic(obs_dim, action_dim)
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2.5e-4)
        self.memory = []
        self.episode_buffer = []
        self.clip = 0.2
        self.gamma = 0.99
        self.lambd = 0.95
        self.name = name
        self.max_steps = max_steps
        self.current_step = 0
        self.last_logprob = None
        self.last_value = None
        self.last_action = None
        self.last_obs = None

    def get_action(self, obs):
        self.last_obs = obs
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = self.model(obs_t)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.last_action = action.item()
        self.last_logprob = dist.log_prob(action)
        self.last_value = value.item()
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.add_reward(0, done=True)
            self.end_episode()
        return self.last_action, self.last_logprob, self.last_value

    def store(self, transition):
        self.memory.append(transition)

    def add_reward(self, reward, done=False):
        self.episode_buffer.append((
            self.last_obs,
            self.last_action,
            self.last_logprob.item(),
            reward,
            self.last_value,
            done
        ))
        if done and self.current_step < self.max_steps:
            self.end_episode()
    def episode_begin(self):
        pass
    def end_episode(self):
        self.memory.extend(self.episode_buffer)
        self.episode_buffer.clear()
        self.current_step = 0
        self.update()
        self.episode_begin()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = list(values) + [0.0]
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lambd * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, epochs=4, batch_size=64):
        if len(self.memory) == 0:
            return

        obs, actions, logprobs, rewards, values, dones = zip(*self.memory)
        advantages = self.compute_gae(rewards, values, dones)
        returns = [a + v for a, v in zip(advantages, values)]

        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions)
        old_logprobs = torch.tensor(logprobs)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        for _ in range(epochs):
            for i in range(0, len(obs), batch_size):
                batch = slice(i, i+batch_size)
                logits, value_preds = self.model(obs[batch])
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                new_logprobs = dist.log_prob(actions[batch])
                ratio = torch.exp(new_logprobs - old_logprobs[batch])

                surr1 = ratio * advantages[batch]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages[batch]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (returns[batch] - value_preds.squeeze()).pow(2).mean()

                self.optimizer.zero_grad()
                (policy_loss + 0.5 * value_loss).backward()
                self.optimizer.step()

        self.memory.clear()