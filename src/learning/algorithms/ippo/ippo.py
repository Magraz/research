import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from learning.algorithms.ippo.models.hybrid_mlp_ac import Hybrid_MLP_AC
from learning.algorithms.ippo.models.mlp_ac import MLP_AC

from learning.environments.types import EnvironmentEnum


class PPOAgent:
    def __init__(
        self,
        env_name,
        state_dim,
        action_dim=None,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        device="cpu",
    ):
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.device = device

        # Initialize network
        if env_name == EnvironmentEnum.BOX2D_SALP:
            self.policy = Hybrid_MLP_AC(state_dim, action_dim).to(device)
            self.policy_old = Hybrid_MLP_AC(state_dim, action_dim).to(device)

        else:
            self.policy = MLP_AC(state_dim, action_dim).to(device)
            self.policy_old = MLP_AC(state_dim, action_dim).to(device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Initialize buffer with empty tensors lists
        self.reset_buffer()

    def reset_buffer(self):
        """Reset the agent's buffer with empty lists to store tensors"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        # This will be determined when the first action is stored
        self.is_dict_action = None

    def get_action(self, state, deterministic=False):
        """Get action from policy while minimizing tensor conversions"""

        state_tensor = torch.FloatTensor(state).to(self.device)

        # Get action, log_prob and value from network
        action, log_prob, value = self.policy_old.act(state_tensor, deterministic)

        # For environment interaction, convert to numpy
        return (
            action.cpu().numpy(),
            log_prob.cpu().item(),
            value.cpu().item(),
        )

    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition in buffer, converting to tensors if needed"""
        # Convert state to tensor and store
        if not torch.is_tensor(state):
            self.states.append(torch.FloatTensor(state).to(self.device))
        else:
            self.states.append(state.to(self.device))

        self.actions.append(torch.FloatTensor(action).to(self.device))

        # Store other transition components as tensors
        self.rewards.append(torch.tensor(reward, dtype=torch.float32).to(self.device))
        self.log_probs.append(
            torch.tensor(log_prob, dtype=torch.float32).to(self.device)
        )
        self.values.append(torch.tensor(value, dtype=torch.float32).to(self.device))
        self.dones.append(torch.tensor(done, dtype=torch.float32).to(self.device))

    def compute_returns_and_advantages(self, next_value=0):
        """Compute returns and advantages using all tensor operations"""
        next_value = torch.tensor(next_value, dtype=torch.float32).to(self.device)

        # Process all data as tensors
        rewards = torch.stack(self.rewards)
        values = torch.cat([torch.stack(self.values), next_value.unsqueeze(0)])
        dones = torch.cat([torch.stack(self.dones), torch.zeros(1, device=self.device)])

        # Initialize advantages tensor
        advantages = torch.zeros_like(rewards)

        # Compute GAE
        gae = torch.tensor(0.0, device=self.device)
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * (1 - dones[step])
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages[step] = gae

        # Compute returns (properly using tensor operations)
        returns = advantages + values[:-1]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def update(self, next_value=0, minibatch_size=128, epochs=10):
        """Update policy with minimal tensor-numpy conversions"""

        # Compute returns and advantages using tensor operations
        returns, advantages = self.compute_returns_and_advantages(next_value)
        returns = returns.unsqueeze(-1)
        advantages = advantages.unsqueeze(-1)

        # Stack states
        states = torch.stack(self.states)
        old_log_probs = torch.stack(self.log_probs).unsqueeze(-1)

        actions = torch.stack(self.actions)
        dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)

        dataloader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

        # Training statistics
        stats = {"total_loss": 0, "policy_loss": 0, "value_loss": 0, "entropy_loss": 0}

        # Multiple epochs of optimization
        for epoch in range(epochs):
            for (
                b_old_states,
                b_old_actions,
                b_old_logprobs,
                b_returns,
                b_advantages,
            ) in dataloader:

                # Forward pass
                log_probs, values, entropy = self.policy.evaluate(
                    b_old_states, b_old_actions
                )

                # PPO objective
                ratio = torch.exp(log_probs - b_old_logprobs)
                surr1 = ratio * b_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * b_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, b_returns)

                # Entropy loss for exploration
                entropy_loss = -self.entropy_coef * entropy.mean()

                # Total loss
                loss = policy_loss + value_loss + entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                # Update statistics
                stats["total_loss"] += loss.item()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy_loss"] += entropy_loss.item()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Reset buffer after update
        self.reset_buffer()

        # Normalize statistics by number of updates
        num_updates = epochs * len(dataloader)
        for key in stats:
            stats[key] /= max(1, num_updates)

        return stats
