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
        action_dim=None,  # Make optional
        action_space=None,  # Add action_space parameter
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

        # Handle both Dict action spaces and regular action spaces
        if action_space is not None:
            self.action_space = action_space
            # For Dict action spaces
            self.network = PPONetwork(state_dim, action_space).to(device)
        else:
            # For legacy Box action spaces
            self.action_dim = action_dim
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

    def get_action(self, state, deterministic=False):
        """
        Get action from the policy with optional deterministic behavior.

        Args:
            state: The current state
            deterministic: If True, use mean action without sampling

        Returns:
            action, log_prob, value
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Check if we have a dict action space
            if hasattr(self, "action_space") and hasattr(self.action_space, "spaces"):
                # For Dict action spaces, use the network's get_action method
                # which should have a deterministic parameter
                return self.network.get_action(
                    state_tensor, deterministic=deterministic
                )
            else:
                # For simple action spaces
                action_mean, action_log_std, value = self.network(state_tensor)

                # Use mean action directly if deterministic
                if deterministic:
                    action = torch.tanh(action_mean)
                else:
                    action_std = torch.exp(action_log_std)
                    normal = torch.distributions.Normal(action_mean, action_std)
                    action = torch.tanh(normal.sample())

                # Calculate log probability
                log_prob = self._calculate_log_prob(action_mean, action_log_std, action)

                # Convert to numpy
                if isinstance(action, dict):
                    # Handle dict actions
                    numpy_action = {}
                    for key, tensor in action.items():
                        numpy_action[key] = tensor.cpu().numpy()[0]
                    return numpy_action, log_prob.cpu().item(), value.cpu().item()
                else:
                    # Handle tensor actions
                    return (
                        action.cpu().numpy()[0],
                        log_prob.cpu().item(),
                        value.cpu().item(),
                    )

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
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Check if we're dealing with Dict actions
        is_dict_action = isinstance(self.actions[0], dict)

        if is_dict_action:
            # Handle dictionary actions
            movement = torch.FloatTensor(
                np.array([a["movement"] for a in self.actions])
            ).to(self.device)
            link_openness = torch.FloatTensor(
                np.array([a["link_openness"] for a in self.actions])
            ).to(self.device)
            detach = torch.FloatTensor(
                np.array([a["detach"] for a in self.actions])
            ).to(self.device)

            # Create a dataset that includes all tensor components
            dataset = TensorDataset(
                states,
                movement,
                link_openness,
                detach,
                old_log_probs,
                returns,
                advantages,
            )
        else:
            # Legacy handling for simple actions
            actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
            dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training statistics
        total_loss = 0
        policy_loss_total = 0
        value_loss_total = 0
        entropy_loss_total = 0

        # Multiple epochs of optimization
        for epoch in range(epochs):
            for batch_data in dataloader:
                if is_dict_action:
                    (
                        batch_states,
                        batch_movement,
                        batch_link_openness,
                        batch_detach,
                        batch_old_log_probs,
                        batch_returns,
                        batch_advantages,
                    ) = batch_data

                    # Reconstruct the dictionary action
                    batch_actions = {
                        "movement": batch_movement,
                        "link_openness": batch_link_openness,
                        "detach": batch_detach,
                    }
                else:
                    # Legacy unpacking
                    (
                        batch_states,
                        batch_actions,
                        batch_old_log_probs,
                        batch_returns,
                        batch_advantages,
                    ) = batch_data

                # Forward pass
                log_probs, values, entropy = self.network.evaluate_action(
                    batch_states, batch_actions
                )

                # The rest of the training loop remains unchanged
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.reshape(-1), batch_returns)

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
