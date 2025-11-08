import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from learning.algorithms.mappo.network import MAPPONetwork


class MAPPOAgent:
    """Multi-Agent PPO with centralized critic"""

    def __init__(
        self,
        observation_dim: int,
        global_state_dim: int,
        action_dim: int,
        n_agents: int,
        params,
        device: str = "cpu",
        discrete: bool = False,
        share_actor: bool = True,
    ):
        self.device = device
        self.n_agents = n_agents
        self.observation_dim = observation_dim
        self.global_state_dim = global_state_dim
        self.discrete = discrete
        self.share_actor = share_actor

        # PPO hyperparameters
        self.gamma = params.gamma
        self.gae_lambda = params.lmbda
        self.clip_epsilon = params.eps_clip
        self.entropy_coef = params.ent_coef
        self.value_coef = params.val_coef
        self.grad_clip = params.grad_clip

        # Create network
        self.network = MAPPONetwork(
            observation_dim=observation_dim,
            global_state_dim=global_state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            discrete=discrete,
            share_actor=share_actor,
        ).to(device)

        # Create old network for PPO
        self.network_old = MAPPONetwork(
            observation_dim=observation_dim,
            global_state_dim=global_state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            discrete=discrete,
            share_actor=share_actor,
        ).to(device)

        self.network_old.load_state_dict(self.network.state_dict())

        # Optimizer for all parameters
        self.optimizer = optim.Adam(self.network.parameters(), lr=params.lr)

        # Buffers for each agent
        self.reset_buffers()

    def reset_buffers(self):
        """Reset experience buffers for all agents"""
        self.observations = [[] for _ in range(self.n_agents)]
        self.global_states = []
        self.actions = [[] for _ in range(self.n_agents)]
        self.rewards = [[] for _ in range(self.n_agents)]
        self.log_probs = [[] for _ in range(self.n_agents)]
        self.values = []  # Shared values from centralized critic
        self.dones = [[] for _ in range(self.n_agents)]

    def get_actions(self, observations, global_state, deterministic=False):
        """
        Get actions for all agents

        Args:
            observations: List of observations, one per agent
            global_state: Concatenated global state (all agent observations)
            deterministic: Whether to use deterministic actions

        Returns:
            actions: List of actions for each agent
            log_probs: List of log probabilities
            value: Single value from centralized critic
        """
        with torch.no_grad():
            # Convert observations to tensors
            obs_tensors = [
                torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                for obs in observations
            ]

            # Convert global state to tensor
            global_state_tensor = (
                torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
            )

            # Get actions from each actor
            actions = []
            log_probs = []

            for agent_idx, obs_tensor in enumerate(obs_tensors):
                action, log_prob = self.network_old.act(
                    obs_tensor, agent_idx, deterministic
                )
                actions.append(action.squeeze(0).cpu().numpy())
                log_probs.append(log_prob.squeeze(0).cpu().item())

            # Get value from centralized critic
            value = self.network_old.get_value(global_state_tensor)
            value = value.squeeze(0).cpu().item()

        return actions, log_probs, value

    def store_transition(
        self, observations, global_state, actions, rewards, log_probs, value, dones
    ):
        """Store transitions for all agents"""
        # Store global state (shared)
        if not torch.is_tensor(global_state):
            self.global_states.append(torch.FloatTensor(global_state).to(self.device))
        else:
            self.global_states.append(global_state.to(self.device))

        # Store value (shared)
        self.values.append(torch.tensor(value, dtype=torch.float32).to(self.device))

        # Store per-agent data
        for agent_idx in range(self.n_agents):
            # Observation
            self.observations[agent_idx].append(
                torch.FloatTensor(observations[agent_idx]).to(self.device)
            )

            # Action
            self.actions[agent_idx].append(
                torch.FloatTensor(actions[agent_idx]).to(self.device)
            )

            # Reward, log_prob, done
            self.rewards[agent_idx].append(
                torch.tensor(rewards[agent_idx], dtype=torch.float32).to(self.device)
            )
            self.log_probs[agent_idx].append(
                torch.tensor(log_probs[agent_idx], dtype=torch.float32).to(self.device)
            )
            self.dones[agent_idx].append(
                torch.tensor(dones[agent_idx], dtype=torch.float32).to(self.device)
            )

    def compute_returns_and_advantages(self, next_value=0):
        """Compute returns and advantages using shared values"""
        if not torch.is_tensor(next_value):
            next_value = torch.tensor(next_value, dtype=torch.float32).to(self.device)

        # Use shared values from centralized critic
        values = torch.cat([torch.stack(self.values).detach(), next_value.unsqueeze(0)])

        # Compute advantages for each agent separately
        all_returns = []
        all_advantages = []

        for agent_idx in range(self.n_agents):
            rewards = torch.stack(self.rewards[agent_idx])
            dones = torch.cat(
                [torch.stack(self.dones[agent_idx]), torch.zeros(1, device=self.device)]
            )

            # Initialize advantages
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

            # Compute returns
            returns = (advantages + values[:-1]).detach()

            all_returns.append(returns)
            all_advantages.append(advantages.detach())

        return all_returns, all_advantages

    def update(self, next_value=0, minibatch_size=128, epochs=10):
        """Update all agents using shared critic"""

        # Compute returns and advantages
        all_returns, all_advantages = self.compute_returns_and_advantages(next_value)

        # Training statistics
        stats = {"total_loss": 0, "policy_loss": 0, "value_loss": 0, "entropy_loss": 0}

        num_updates = 0

        # Update each agent (or all at once if sharing actor)
        if self.share_actor:
            # Combine all agent data for shared actor update
            all_obs = []
            all_global_states = []
            all_actions = []
            all_old_log_probs = []
            all_returns_combined = []
            all_advantages_combined = []

            for agent_idx in range(self.n_agents):
                # Normalize advantages for this agent
                advantages = all_advantages[agent_idx]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Stack data
                all_obs.append(torch.stack(self.observations[agent_idx]))
                all_actions.append(torch.stack(self.actions[agent_idx]))
                all_old_log_probs.append(torch.stack(self.log_probs[agent_idx]))
                all_returns_combined.append(all_returns[agent_idx])
                all_advantages_combined.append(advantages)

            # Repeat global states for each agent
            global_states_stacked = torch.stack(self.global_states)
            all_global_states = global_states_stacked.repeat(self.n_agents, 1)

            # Concatenate all agent data
            combined_obs = torch.cat(all_obs, dim=0).detach()
            combined_actions = torch.cat(all_actions, dim=0).detach()
            combined_old_log_probs = torch.cat(all_old_log_probs, dim=0).detach()
            combined_returns = torch.cat(all_returns_combined, dim=0)
            combined_advantages = torch.cat(all_advantages_combined, dim=0)

            # Create dataset
            dataset = TensorDataset(
                combined_obs,
                all_global_states,
                combined_actions,
                combined_old_log_probs,
                combined_returns,
                combined_advantages,
            )

            dataloader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

            # Train for multiple epochs
            for epoch in range(epochs):
                for batch in dataloader:
                    (
                        batch_obs,
                        batch_global_states,
                        batch_actions,
                        batch_old_log_probs,
                        batch_returns,
                        batch_advantages,
                    ) = batch

                    # We need to know which agent each sample belongs to
                    # For shared actor, we can use agent_idx = 0 (same for all)
                    log_probs, values, entropy = self.network.evaluate_actions(
                        batch_obs, batch_global_states, batch_actions, agent_idx=0
                    )

                    # PPO objective
                    ratio = torch.exp(log_probs.squeeze(-1) - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = (
                        torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                        * batch_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = F.mse_loss(values, batch_returns)

                    # Entropy loss
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
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.grad_clip
                    )
                    self.optimizer.step()

                    # Update statistics
                    stats["total_loss"] += loss.item()
                    stats["policy_loss"] += policy_loss.item()
                    stats["value_loss"] += value_loss.item()
                    stats["entropy_loss"] += entropy_loss.item()
                    num_updates += 1

        else:
            # Update each agent separately (independent actors)
            for agent_idx in range(self.n_agents):
                # Normalize advantages
                advantages = all_advantages[agent_idx]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                returns = all_returns[agent_idx]

                # Stack data
                obs = torch.stack(self.observations[agent_idx]).detach()
                global_states = torch.stack(self.global_states).detach()
                actions = torch.stack(self.actions[agent_idx]).detach()
                old_log_probs = torch.stack(self.log_probs[agent_idx]).detach()

                dataset = TensorDataset(
                    obs, global_states, actions, old_log_probs, returns, advantages
                )
                dataloader = DataLoader(
                    dataset, batch_size=minibatch_size, shuffle=True
                )

                # Train for multiple epochs
                for epoch in range(epochs):
                    for batch in dataloader:
                        (
                            batch_obs,
                            batch_global_states,
                            batch_actions,
                            batch_old_log_probs,
                            batch_returns,
                            batch_advantages,
                        ) = batch

                        # Forward pass
                        log_probs, values, entropy = self.network.evaluate_actions(
                            batch_obs, batch_global_states, batch_actions, agent_idx
                        )

                        # PPO objective
                        ratio = torch.exp(log_probs.squeeze(-1) - batch_old_log_probs)
                        surr1 = ratio * batch_advantages
                        surr2 = (
                            torch.clamp(
                                ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                            )
                            * batch_advantages
                        )
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Value loss
                        value_loss = F.mse_loss(values, batch_returns)

                        # Entropy loss
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
                        torch.nn.utils.clip_grad_norm_(
                            self.network.parameters(), self.grad_clip
                        )
                        self.optimizer.step()

                        # Update statistics
                        stats["total_loss"] += loss.item()
                        stats["policy_loss"] += policy_loss.item()
                        stats["value_loss"] += value_loss.item()
                        stats["entropy_loss"] += entropy_loss.item()
                        num_updates += 1

        # Update old network
        self.network_old.load_state_dict(self.network.state_dict())

        # Reset buffers
        self.reset_buffers()

        # Average statistics
        for key in stats:
            stats[key] /= max(1, num_updates)

        return stats
