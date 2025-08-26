import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from learning.algorithms.ippo.network import PPONetwork


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        device="cpu",
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device

        # Create network and optimizer
        self.network = PPONetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Storage for trajectory data
        self.reset_buffer()

    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def get_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.network.get_action(state_tensor)

            return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item()

    def store_transition(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(self, next_value=0):
        returns = []
        advantages = []

        # Convert to numpy arrays
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones + [False])

        # Compute advantages using GAE
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * (1 - dones[step])
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)

        # Compute returns
        returns = [adv + val for adv, val in zip(advantages, self.values)]

        return np.array(returns), np.array(advantages)

    def update(self, next_value=0, epochs=10, batch_size=64):
        if len(self.states) == 0:
            return {}

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Create a dataset and DataLoader
        dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training statistics
        total_loss = 0
        policy_loss_total = 0
        value_loss_total = 0
        entropy_loss_total = 0

        # Multiple epochs of optimization
        for epoch in range(epochs):
            for (
                batch_states,
                batch_actions,
                batch_old_log_probs,
                batch_returns,
                batch_advantages,
            ) in dataloader:
                # Forward pass
                log_probs, values, entropy = self.network.evaluate_action(
                    batch_states, batch_actions
                )

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                # Accumulate losses
                total_loss += loss.item()
                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                entropy_loss_total += entropy_loss.item()

        # Reset storage
        self.reset_buffer()

        num_updates = epochs * len(dataloader)
        return {
            "total_loss": total_loss / num_updates,
            "policy_loss": policy_loss_total / num_updates,
            "value_loss": value_loss_total / num_updates,
            "entropy_loss": entropy_loss_total / num_updates,
        }
