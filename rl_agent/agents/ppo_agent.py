import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import os
from typing import List

# Ensure the BaseAgent is importable from its location
from .base_agent import BaseAgent

class TrajectoryBuffer:
    """A buffer to store trajectories (sequences of state, action, etc.)."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminal = []

    def clear(self):
        """Clears all stored trajectories."""
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminal[:]

    def add(self, state, action, reward, done, log_prob):
        """Adds a new experience to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_terminal.append(done)
        self.log_probs.append(log_prob)

class ActorCritic(nn.Module):
    """
    An Actor-Critic network for PPO.
    It has two separate output heads for the two-part discrete action, which is
    necessary for the MultiDiscrete action space.
    """
    def __init__(self, state_dim: int, action_dims: List[int]):
        super(ActorCritic, self).__init__()

        # Shared layers to process the state
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )

        # Actor heads - one for each part of the MultiDiscrete action
        self.actor_head1 = nn.Linear(128, action_dims[0])
        self.actor_head2 = nn.Linear(128, action_dims[1])

        # Critic head for state-value estimation
        self.critic_head = nn.Linear(128, 1)

    def forward(self, state):
        """
        Forward pass through the network.
        Returns logits for each action head and the state value.
        """
        # Add a batch dimension if the state is a single vector
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = self.shared_layers(state)
        action_logits1 = self.actor_head1(x)
        action_logits2 = self.actor_head2(x)
        state_value = self.critic_head(x)
        return action_logits1, action_logits2, state_value

class PPOAgent(BaseAgent):
    """PPO Agent that handles the MultiDiscrete action space for gear generation."""
    
    def __init__(self, state_dim: int, action_dims: List[int], lr: float, 
                 gamma: float, clip_epsilon: float, epochs: int = 10):
        
        super().__init__(state_dim, action_dims) 
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        
        self.policy = ActorCritic(state_dim, action_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(state_dim, action_dims).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.memory = TrajectoryBuffer()
        self.mse_loss = nn.MSELoss()

    def act(self, state, deterministic=False):
        """Selects an action using the old policy."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            # The forward pass returns logits for both action heads
            logits1, logits2, _ = self.policy_old(state)
            
            dist1 = Categorical(logits=logits1)
            dist2 = Categorical(logits=logits2)
            
            if deterministic:
                # For evaluation, pick the most likely action (best move)
                action1 = dist1.probs.argmax(dim=-1)
                action2 = dist2.probs.argmax(dim=-1)
            else:
                # For training, sample from the distribution to explore
                action1 = dist1.sample()
                action2 = dist2.sample()
            
            log_prob = dist1.log_prob(action1) + dist2.log_prob(action2)
            # Stack the two actions into a single tensor
            action = torch.stack([action1, action2], dim=-1)

        # Squeeze to remove batch dim, convert to numpy, and return log_prob as a float
        return action.squeeze(0).cpu().numpy(), log_prob.item()

    def update(self):
        """Updates the policy network using data from the memory buffer."""
        rewards_to_go = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminal)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_to_go.insert(0, discounted_reward)

        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32).to(self.device)
        old_states = torch.tensor(np.array(self.memory.states), dtype=torch.float32).to(self.device)
        old_actions = torch.tensor(np.array(self.memory.actions), dtype=torch.int64).to(self.device)
        old_log_probs = torch.tensor(self.memory.log_probs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            _, _, state_values = self.policy(old_states)
            state_values = torch.squeeze(state_values)
        advantages = rewards_to_go - state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for _ in range(self.epochs):
            logits1, logits2, state_values = self.policy(old_states)
            state_values = torch.squeeze(state_values)
            
            dist1 = Categorical(logits=logits1)
            dist2 = Categorical(logits=logits2)
            
            log_probs1 = dist1.log_prob(old_actions[:, 0])
            log_probs2 = dist2.log_prob(old_actions[:, 1])
            new_log_probs = log_probs1 + log_probs2
            entropy = dist1.entropy() + dist2.entropy()

            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.mse_loss(state_values, rewards_to_go) - 0.01 * entropy.mean()
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()

        return total_loss / self.epochs

    def save(self, path):
        """Saves the model's state dictionary."""
        print(f"Saving model to {path}")
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        """Loads the model's state dictionary."""
        print(f"Loading model from {path}")
        # <--- FIX: Added weights_only=True to suppress warning ---
        self.policy.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.policy_old.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        # <--- END OF FIX ---

