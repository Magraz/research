import torch
import numpy as np


class TensorRolloutBuffer:
    """Pre-allocated tensor-based rollout buffer for efficient parallel collection."""

    def __init__(self, time_steps, n_envs, n_agents, d_state, d_action, device):
        self.device = device

        # Pre-allocate tensors for the whole rollout
        self.states = torch.zeros(
            (time_steps, n_envs, n_agents, d_state), device=device
        )
        self.actions = torch.zeros(
            (time_steps, n_envs, n_agents, d_action), device=device
        )
        self.logprobs = torch.zeros((time_steps, n_envs, n_agents), device=device)
        self.rewards = torch.zeros((time_steps, n_envs), device=device)
        self.values = torch.zeros((time_steps, n_envs, 1), device=device)
        self.is_terminals = torch.zeros(
            (time_steps, n_envs), dtype=torch.bool, device=device
        )

        # Track the current step in the buffer
        self.step = 0

    def add(self, t, state, action, log_prob, value, reward, done):
        """Add a transition at timestep t."""
        self.states[t] = state
        self.actions[t] = action
        self.logprobs[t] = log_prob
        self.values[t] = value
        self.rewards[t] = reward
        self.is_terminals[t] = done


class EnvRunner:
    """Runs a batch of environments for collecting experience."""

    def __init__(self, policy, env, device):
        self.env = env
        self.policy = policy
        self.device = device

    def run(self, steps):
        """Collect 'steps' timesteps of experience from the environment."""
        # Get environment dimensions
        n_envs = self.env.num_envs
        n_agents = self.env.n_agents
        d_state = self.policy.d_state
        d_action = self.policy.d_action

        # Create buffer to store rollout data
        buffer = TensorRolloutBuffer(
            steps, n_envs, n_agents, d_state, d_action, self.device
        )

        # Initial reset
        states = self.env.reset()

        # Process initial state (VMAS specific processing)
        processed_states = self._process_state(states)

        # Collect data for steps timesteps
        for t in range(steps):
            # Get actions from policy
            with torch.no_grad():
                actions, log_probs, values = self.policy.act(processed_states)

            # Step the environment
            # For VMAS, prepare actions in the format it expects
            action_tensor_list = self._prepare_actions_for_vmas(actions)
            next_states, rewards, dones, _ = self.env.step(action_tensor_list)

            # Store transition in buffer
            buffer.add(t, processed_states, actions, log_probs, values, rewards, dones)

            # Handle episode terminations
            if dones.any():
                done_indices = torch.where(dones)[0]
                for idx in done_indices:
                    # Reset terminated environments
                    next_states = self._update_states_after_reset(next_states, idx)

            # Update for next step
            states = next_states
            processed_states = self._process_state(states)

        buffer.step = steps  # Mark that the buffer is full
        return buffer

    def _process_state(self, states):
        """Process raw environment states into the format expected by the policy."""
        # This depends on how your VMAS environment returns states and what your policy expects
        # Example implementation (adjust based on your specific needs):
        from learning.algorithms.ppo.utils import process_state

        # Assuming your VMAS returns a dictionary or list that needs to be processed
        if isinstance(states, dict):
            # Example: process dictionary observations from VMAS
            processed = torch.cat(
                [states[key] for key in sorted(states.keys())], dim=-1
            )
        elif isinstance(states, list):
            # Example: process list observations from VMAS
            processed = torch.stack(states, dim=1)  # stack along agent dimension
        else:
            # Already a tensor in the right format
            processed = states

        return processed.to(self.device)

    def _prepare_actions_for_vmas(self, actions):
        """Format actions for VMAS environment step method."""
        # Permute action tensor of shape (n_envs, n_agents * action_dim) to (agents, n_env, action_dim)
        action_tensor = actions.reshape(
            self.env.num_envs,
            self.env.n_agents,
            self.policy.d_action,
        ).transpose(1, 0)

        # Turn action tensor into list of tensors with shape (n_env, action_dim)
        action_tensor_list = torch.unbind(action_tensor)

        return action_tensor_list

    def _update_states_after_reset(self, states, reset_idx):
        """Update states after resetting a specific environment."""
        # Reset the specific environment
        reset_states = self.env.reset_at(index=reset_idx)

        # Update the states based on the structure
        if isinstance(states, dict):
            for key in states:
                states[key][reset_idx] = reset_states[key]
        elif isinstance(states, list):
            for i in range(len(states)):
                states[i][reset_idx] = reset_states[i]
        else:
            # Tensor case
            states[reset_idx] = reset_states

        return states
