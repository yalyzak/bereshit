import copy
import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim_discrete = 0, action_dim_continuous=0):
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
    def __deepcopy__(self, memo):
        # Create a new blank Agent object (skip __init__)
        new_agent = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_agent

        # Copy everything normally, except model and optimizer
        for k, v in self.__dict__.items():
            if k in ('model', 'optimizer'):
                setattr(new_agent, k, v)  # shallow copy (pointer)
            else:
                setattr(new_agent, k, copy.deepcopy(v, memo))

        return new_agent
    def __init__(self, obs_dim, action_dim_discrete=0,action_dim_continuous=0, name="Agent", max_steps=100,optimizer=None,model=None):
        if model is not None:
            self.model = model
        else:
            self.model = ActorCritic(obs_dim, action_dim_discrete=action_dim_discrete,action_dim_continuous=action_dim_continuous)
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
        self.last_discrete_logprob = None
        self.last_continuous_logprob = None
        self.last_value = None
        self.last_discrete_action = None
        self.last_continuous_action = None

        self.last_obs = None
        self.ready_to_update = False
        self.done_reward = 0

    def get_discrete_action(self, obs):
        self.last_obs = obs

        # Convert observation to a tensor with batch dimension
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # Forward pass
        logits, _, _, value = self.model(obs_t)

        # Compute probabilities
        probs = torch.softmax(logits, dim=-1)

        # Create categorical distribution
        dist = torch.distributions.Categorical(probs)

        # Sample discrete action
        action = dist.sample()

        # Store for later use (e.g., PPO update)
        self.last_discrete_action = action.item()
        self.last_discrete_logprob = dist.log_prob(action).detach()
        self.last_value = value.item()

        # Step management
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.add_reward(0, done=True)
            self.end_episode()

        return self.last_discrete_action, self.last_discrete_logprob, self.last_value

    def get_continuous_action(self, obs):
        if self.model.action_dim_continuous == 0:
            raise RuntimeError("This agent has no continuous action head.")

        self.last_obs = obs

        # Convert observation to a tensor with batch dimension
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # Forward pass
        _, mean, log_std, value = self.model(obs_t)

        # Compute std
        std = log_std.exp()

        # Create Normal distribution
        dist = torch.distributions.Normal(mean, std)

        # Sample continuous action
        action = dist.sample()

        # Store for later use (PPO update)
        self.last_continuous_action = action.squeeze(0).detach().cpu().numpy()
        self.last_continuous_logprob = dist.log_prob(action).sum(-1).detach().item()
        self.last_value = value.item()

        # Step management
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.add_reward(0, done=True)
            self.end_episode()

        return self.last_continuous_action, self.last_continuous_logprob, self.last_value

    def store(self, transition):
        self.memory.append(transition)

    def add_reward(self, reward, done=False):
        self.episode_buffer.append((
            self.last_obs,
            self.last_discrete_action if self.model.action_dim_discrete > 0 else None,
            self.last_continuous_action if self.model.action_dim_continuous > 0 else None,
            self.last_discrete_logprob.item() if self.model.action_dim_discrete > 0 else None,
            self.last_continuous_logprob if self.model.action_dim_continuous > 0 else None,
            reward,
            self.last_value,
            done
        ))

        if done:
            self.done_reward = reward  # record final reward
            self.ready_to_update = True  # set flag for main thread

    def On_episode_begin(self):
        g= False
        for component in self.parent.components.values():
            if hasattr(component, 'On_episode_begin') and component.On_episode_begin is not None and component != self:
                component.On_episode_begin()
                g= True
        if not g:
            print(f"{self.name} episode_begin was not define")
            pass
    def end_episode(self):
        self.memory.extend(self.episode_buffer)
        self.episode_buffer.clear()
        self.current_step = 0
        self.update()
        self.On_episode_begin()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = [v.detach() if hasattr(v, 'detach') else v for v in values] + [0.0]

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lambd * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, epochs=4, batch_size=64):
        if len(self.memory) == 0:
            return

        # Unpack memory
        (
            obs,
            discrete_actions,
            continuous_actions,
            discrete_logprobs,
            continuous_logprobs,
            rewards,
            values,
            dones
        ) = zip(*self.memory)

        advantages = self.compute_gae(rewards, values, dones)
        returns = [a + v for a, v in zip(advantages, values)]

        # Convert common tensors
        obs = torch.tensor(obs, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        has_discrete = self.model.action_dim_discrete > 0
        has_continuous = self.model.action_dim_continuous > 0

        if has_discrete:
            discrete_actions = torch.tensor(discrete_actions)
            old_discrete_logprobs = torch.tensor(discrete_logprobs, dtype=torch.float32)

        if has_continuous:
            continuous_actions = torch.from_numpy(np.array(continuous_actions)).float()
            old_continuous_logprobs = torch.tensor(continuous_logprobs, dtype=torch.float32)

        for _ in range(epochs):
            for i in range(0, len(obs), batch_size):
                batch = slice(i, i + batch_size)

                logits, mean, log_std, value_preds = self.model(obs[batch])

                policy_loss = 0.0

                # Discrete policy loss
                if has_discrete:
                    probs = torch.softmax(logits, dim=-1)
                    dist_discrete = torch.distributions.Categorical(probs)
                    new_discrete_logprobs = dist_discrete.log_prob(discrete_actions[batch])
                    ratio_discrete = torch.exp(new_discrete_logprobs - old_discrete_logprobs[batch])
                    surr1_discrete = ratio_discrete * advantages[batch]
                    surr2_discrete = torch.clamp(ratio_discrete, 1 - self.clip, 1 + self.clip) * advantages[batch]
                    policy_loss_discrete = -torch.min(surr1_discrete, surr2_discrete).mean()
                    policy_loss = policy_loss + policy_loss_discrete

                # Continuous policy loss
                if has_continuous:
                    std = log_std.exp()
                    dist_continuous = torch.distributions.Normal(mean, std)
                    new_continuous_logprobs = dist_continuous.log_prob(continuous_actions[batch]).sum(-1)
                    ratio_continuous = torch.exp(new_continuous_logprobs - old_continuous_logprobs[batch])
                    surr1_continuous = ratio_continuous * advantages[batch]
                    surr2_continuous = torch.clamp(ratio_continuous, 1 - self.clip, 1 + self.clip) * advantages[batch]
                    policy_loss_continuous = -torch.min(surr1_continuous, surr2_continuous).mean()
                    policy_loss = policy_loss + policy_loss_continuous

                # Value loss
                value_loss = (returns[batch] - value_preds.squeeze(-1)).pow(2).mean()

                total_loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        self.memory.clear()

        if len(self.memory) == 0:
            return

        # Unpack memory
        (
            obs,
            discrete_actions,
            continuous_actions,
            discrete_logprobs,
            continuous_logprobs,
            rewards,
            values,
            dones
        ) = zip(*self.memory)

        advantages = self.compute_gae(rewards, values, dones)
        returns = [a + v for a, v in zip(advantages, values)]

        # Convert everything to tensors
        obs = torch.tensor(obs, dtype=torch.float32)
        discrete_actions = torch.tensor(discrete_actions)
        continuous_actions = torch.from_numpy(np.array(continuous_actions)).float()
        old_discrete_logprobs = torch.tensor(discrete_logprobs, dtype=torch.float32)
        old_continuous_logprobs = torch.tensor(continuous_logprobs, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        for _ in range(epochs):
            for i in range(0, len(obs), batch_size):
                batch = slice(i, i + batch_size)

                # Forward pass
                logits, mean, log_std, value_preds = self.model(obs[batch])

                # Discrete distribution
                probs = torch.softmax(logits, dim=-1)
                dist_discrete = torch.distributions.Categorical(probs)
                new_discrete_logprobs = dist_discrete.log_prob(discrete_actions[batch])

                # Continuous distribution
                std = log_std.exp()
                dist_continuous = torch.distributions.Normal(mean, std)
                new_continuous_logprobs = dist_continuous.log_prob(continuous_actions[batch]).sum(-1)

                # Discrete ratio
                ratio_discrete = torch.exp(new_discrete_logprobs - old_discrete_logprobs[batch])
                # Continuous ratio
                ratio_continuous = torch.exp(new_continuous_logprobs - old_continuous_logprobs[batch])

                # Surrogate objectives
                surr1_discrete = ratio_discrete * advantages[batch]
                surr2_discrete = torch.clamp(ratio_discrete, 1 - self.clip, 1 + self.clip) * advantages[batch]
                policy_loss_discrete = -torch.min(surr1_discrete, surr2_discrete).mean()

                surr1_continuous = ratio_continuous * advantages[batch]
                surr2_continuous = torch.clamp(ratio_continuous, 1 - self.clip, 1 + self.clip) * advantages[batch]
                policy_loss_continuous = -torch.min(surr1_continuous, surr2_continuous).mean()

                # Combine policy losses (you can weight if you prefer)
                policy_loss = policy_loss_discrete + policy_loss_continuous

                # Value loss
                value_loss = (returns[batch] - value_preds.squeeze(-1)).pow(2).mean()

                # Total loss
                total_loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        self.memory.clear()

    def attach(self, owner_object):
        pass
