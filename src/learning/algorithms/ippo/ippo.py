import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from learning.algorithms.ippo.network import PPONetwork


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim=None,
        action_space=None,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        device="cpu",
    ):
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device

        # Initialize network
        if action_space is not None:
            self.action_space = action_space
            self.network = PPONetwork(state_dim, action_space).to(device)
        else:
            self.action_dim = action_dim
            self.network = PPONetwork(state_dim, action_dim).to(device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

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
        with torch.no_grad():
            # Convert state to tensor if needed
            if not torch.is_tensor(state):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                # If already tensor, ensure it has batch dimension and is on right device
                state_tensor = (
                    state.unsqueeze(0).to(self.device)
                    if state.dim() == 1
                    else state.to(self.device)
                )

            # Get action, log_prob and value from network
            action, log_prob, value = self.network.get_action(
                state_tensor, deterministic
            )

            # For environment interaction, convert to numpy
            if isinstance(action, dict):
                # Handle dictionary actions
                numpy_action = {}
                for key, tensor in action.items():
                    numpy_action[key] = tensor.cpu().numpy()[0]
                return numpy_action, log_prob.cpu().item(), value.cpu().item()
            else:
                # Handle regular actions
                return (
                    action.cpu().numpy()[0],
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

        # Handle action based on type (first determine action type if not yet set)
        if self.is_dict_action is None:
            self.is_dict_action = isinstance(action, dict)

        if self.is_dict_action:
            # Convert dict action components to tensors
            action_dict = {}
            for key, val in action.items():
                if not torch.is_tensor(val):
                    action_dict[key] = (
                        torch.FloatTensor([val]).to(self.device)
                        if np.isscalar(val)
                        else torch.FloatTensor(val).to(self.device)
                    )
                else:
                    action_dict[key] = val.to(self.device)
            self.actions.append(action_dict)
        else:
            # Convert regular action to tensor
            if not torch.is_tensor(action):
                self.actions.append(torch.FloatTensor(action).to(self.device))
            else:
                self.actions.append(action.to(self.device))

        # Store other transition components as tensors
        self.rewards.append(torch.tensor(reward, dtype=torch.float32).to(self.device))
        self.log_probs.append(
            torch.tensor(log_prob, dtype=torch.float32).to(self.device)
        )
        self.values.append(torch.tensor(value, dtype=torch.float32).to(self.device))
        self.dones.append(torch.tensor(done, dtype=torch.float32).to(self.device))

    def compute_returns_and_advantages(self, next_value=0):
        """Compute returns and advantages using all tensor operations"""
        # Convert next_value to tensor if needed
        if not torch.is_tensor(next_value):
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

        return returns, advantages

    def update(self, next_value=0, epochs=10, batch_size=64):
        """Update policy with minimal tensor-numpy conversions"""
        if len(self.states) == 0:
            return {}

        # Compute returns and advantages using tensor operations
        returns, advantages = self.compute_returns_and_advantages(next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Stack states
        states = torch.stack(self.states)
        old_log_probs = torch.stack(self.log_probs)

        # Handle different action types
        if self.is_dict_action:
            # For dict actions, create separate tensors for each action component
            action_tensors = {}
            for key in self.actions[0].keys():
                # Stack tensors for each component
                try:
                    action_tensors[key] = torch.stack([a[key] for a in self.actions])
                except:
                    # If shapes don't match (e.g., some actions have different dimensions)
                    # Process them individually
                    component_tensors = []
                    for a in self.actions:
                        if a[key].dim() == 0:  # If scalar
                            component_tensors.append(a[key].unsqueeze(0))
                        else:
                            component_tensors.append(a[key])
                    action_tensors[key] = torch.stack(component_tensors)

            # Create dataset with all components
            dataset_tensors = [states, old_log_probs, returns, advantages]
            for key in action_tensors:
                dataset_tensors.insert(1, action_tensors[key])  # Insert after states

            dataset = TensorDataset(*dataset_tensors)

        else:
            # For regular actions, simply stack them
            actions = torch.stack(self.actions)
            dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training statistics
        stats = {"total_loss": 0, "policy_loss": 0, "value_loss": 0, "entropy_loss": 0}

        # Multiple epochs of optimization
        for epoch in range(epochs):
            for batch in dataloader:
                # Process batch based on action type
                if self.is_dict_action:
                    batch_states = batch[0]
                    batch_old_log_probs = batch[-3]
                    batch_returns = batch[-2]
                    batch_advantages = batch[-1]

                    # Extract action components from batch
                    batch_actions = {}
                    for i, key in enumerate(action_tensors.keys()):
                        batch_actions[key] = batch[i + 1]  # +1 because states is first
                else:
                    (
                        batch_states,
                        batch_actions,
                        batch_old_log_probs,
                        batch_returns,
                        batch_advantages,
                    ) = batch

                # Forward pass
                log_probs, values, entropy = self.network.evaluate_action(
                    batch_states, batch_actions
                )

                # PPO objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                # Entropy loss for exploration
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                # Update statistics
                stats["total_loss"] += loss.item()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy_loss"] += entropy_loss.item()

        # Reset buffer after update
        self.reset_buffer()

        # Normalize statistics by number of updates
        num_updates = epochs * len(dataloader)
        for key in stats:
            stats[key] /= max(1, num_updates)

        return stats
