import copy
import os

import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim_discrete=0, action_dim_continuous=0):
        super().__init__()
        self.action_dim_discrete = action_dim_discrete
        self.action_dim_continuous = action_dim_continuous

        # Shared trunk
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        # Discrete action head
        if action_dim_discrete > 0:
            self.actor_discrete = nn.Linear(64, action_dim_discrete)
        else:
            self.actor_discrete = None

        # Continuous action head (mean)
        if action_dim_continuous > 0:
            self.actor_continuous_mean = nn.Linear(64, action_dim_continuous)
            # Log std is learned as a parameter
            self.actor_continuous_log_std = nn.Parameter(torch.zeros(action_dim_continuous))
        else:
            self.actor_continuous_mean = None
            self.actor_continuous_log_std = None

        # Critic head
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        hidden = self.fc(x)

        logits = None
        mean = None
        log_std = None

        if self.actor_discrete is not None:
            logits = self.actor_discrete(hidden)

        if self.actor_continuous_mean is not None:
            # squash to [-1, +1]
            mean = torch.tanh(self.actor_continuous_mean(hidden))
            log_std = self.actor_continuous_log_std.expand_as(mean)

        value = self.critic(hidden)

        return logits, mean, log_std, value


class Agent:

    # 🔹 Shared across ALL agents
    shared_memory = []
    updating = False
    episode = 0

    def __deepcopy__(self, memo):
        new_agent = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_agent

        for k, v in self.__dict__.items():
            if k in ('model', 'optimizer'):
                setattr(new_agent, k, v)  # share
            else:
                setattr(new_agent, k, copy.deepcopy(v, memo))

        return new_agent

    def __init__(
        self,
        obs_dim,
        action_dim_discrete=0,
        action_dim_continuous=0,
        name="Agent",
        max_steps=200,
        optimizer=None,
        model=None,
        save_dir=None,
        update_after=2048
    ):

        self.name = name
        self.max_steps = max_steps
        self.current_step = 0
        self.update_after = update_after

        # 🔹 model
        if model is not None and not isinstance(model, str):
            self.model = model
        else:
            from bereshit.addons.PPO import ActorCritic
            self.model = ActorCritic(
                obs_dim,
                action_dim_discrete,
                action_dim_continuous
            )

        # 🔹 load
        if isinstance(model, str) and os.path.exists(model):
            self.model.load_state_dict(torch.load(model))
            print("✅ Loaded model")

        # 🔹 optimizer (shared)
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=2.5e-4)

        # 🔹 storage
        self.episode_buffer = []
        self.memory = Agent.shared_memory  # 🔥 shared

        # PPO params
        self.clip = 0.2
        self.gamma = 0.99
        self.lambd = 0.95

        # temp
        self.last_obs = None
        self.last_action = None
        self.last_logprob = None
        self.last_value = None

        self.dir = os.path.join(save_dir, "model.pth") if save_dir else None

    # =========================
    # ACTION
    # =========================
    def get_continuous_action(self, obs):
        self.last_obs = obs

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        _, mean, log_std, value = self.model(obs_t)

        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        action = dist.sample()

        self.last_action = action.squeeze(0).detach().numpy()
        self.last_logprob = dist.log_prob(action).sum(-1).item()
        self.last_value = value.item()

        self.current_step += 1

        if self.current_step >= self.max_steps:
            self.add_reward(0, done=True)
            self.end_episode()

        return self.last_action, self.last_logprob, self.last_value

    # =========================
    # STORE
    # =========================
    def add_reward(self, reward, done=False):
        self.episode_buffer.append((
            self.last_obs,
            self.last_action,
            self.last_logprob,
            reward,
            self.last_value,
            done
        ))

        if done:
            self.end_episode()

    # =========================
    # EPISODE END
    # =========================
    def end_episode(self):
        Agent.episode += 1

        self.memory.extend(self.episode_buffer)
        self.episode_buffer.clear()
        self.current_step = 0

        # 🔥 central update trigger
        if len(self.memory) >= self.update_after and not Agent.updating:
            Agent.updating = True
            self.update()
            Agent.updating = False

        self.On_episode_begin()

    def On_episode_begin(self):
        if hasattr(self, "parent"):
            for component in self.parent.components.values():
                if hasattr(component, 'OnEpisodeBegin'):
                    component.OnEpisodeBegin()
    # =========================
    # GAE
    # =========================
    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = list(values) + [0]

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lambd * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        return advantages

    # =========================
    # UPDATE (PPO)
    # =========================
    def update(self, epochs=4, batch_size=64):

        if len(self.memory) == 0:
            return

        obs, actions, logprobs, rewards, values, dones = zip(*self.memory)

        advantages = self.compute_gae(rewards, values, dones)
        returns = [a + v for a, v in zip(advantages, values)]

        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        old_logprobs = torch.tensor(logprobs, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        for _ in range(epochs):
            for i in range(0, len(obs), batch_size):

                batch = slice(i, i + batch_size)

                _, mean, log_std, values_pred = self.model(obs[batch])

                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)

                new_logprobs = dist.log_prob(actions[batch]).sum(-1)

                ratio = torch.exp(new_logprobs - old_logprobs[batch])

                surr1 = ratio * advantages[batch]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages[batch]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (returns[batch] - values_pred.squeeze()).pow(2).mean()

                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # 🔥 clear shared memory AFTER update
        self.memory.clear()

        # 🔹 save
        if self.dir:
            torch.save(self.model.state_dict(), self.dir)
            print(f"💾 Model saved at episode {Agent.episode}")
