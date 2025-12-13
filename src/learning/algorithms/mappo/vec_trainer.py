import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle

from learning.algorithms.mappo.mappo import MAPPOAgent
from learning.environments.types import EnvironmentEnum
from learning.algorithms.env_wrapper import EnvWrapper
from learning.algorithms.create_env import make_vec_env

import torch


class VecMAPPOTrainer:
    def __init__(
        self,
        env,
        env_name,
        n_agents,
        observation_dim,
        global_state_dim,
        action_dim,
        params,
        dirs=None,
        device="cpu",
        share_actor=True,
        n_parallel_envs=4,
    ):
        self.device = device
        self.dirs = dirs
        self.n_agents = n_agents
        self.params = params
        self.env_name = env_name
        self.n_parallel_envs = n_parallel_envs

        # Create environment for evaluation
        self.wrapped_env = EnvWrapper(
            env=env, env_name=env_name, n_agents=self.n_agents
        )

        # Create vectorized environment using Gymnasium's API
        self.vec_env = make_vec_env(
            self.env_name,
            self.n_agents,
            self.n_parallel_envs,
            use_async=True,  # Use parallel processing
        )

        # Set action bounds based on environment
        if env_name in [EnvironmentEnum.MPE_SPREAD, EnvironmentEnum.MPE_SIMPLE]:
            self.discrete = True
        else:
            self.discrete = False

        # Create MAPPO agent
        self.agent = MAPPOAgent(
            observation_dim,
            global_state_dim,
            action_dim,
            self.n_agents,
            self.params,
            self.device,
            self.discrete,
            share_actor,
            self.n_parallel_envs,
        )

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_stats = defaultdict(list)

    def collect_trajectory(self, max_steps):
        """
        Collect trajectory using Gymnasium's vectorized environments.

        Gymnasium VectorEnv provides:
        - obs: (n_envs, *obs_shape)
        - reward: (n_envs,)
        - terminated: (n_envs,)
        - truncated: (n_envs,)
        - info: dict with (n_envs,) arrays
        """

        # Reset environment
        obs, infos = self.vec_env.reset()

        total_step_count = 0
        episode_count = 0
        steps_per_episode = []
        current_episode_steps = np.zeros(self.n_parallel_envs, dtype=np.int32)

        for step in range(max_steps):
            batch_size = obs.shape[0]

            # Construct global states for each environment
            # Shape: (n_envs, n_agents * obs_dim)
            global_states = obs.reshape(batch_size, -1)

            # Get actions for all agents in all environments
            all_actions = []
            all_log_probs = []
            all_values = []

            with torch.no_grad():
                for env_idx in range(batch_size):
                    # Get observations for this environment
                    env_obs = obs[env_idx]  # (n_agents, obs_dim)
                    env_global_state = global_states[env_idx]  # (n_agents * obs_dim)

                    # Get actions for all agents in this environment
                    actions, log_probs, value = self.agent.get_actions(
                        env_obs, env_global_state, deterministic=False
                    )

                    all_actions.append(actions)
                    all_log_probs.append(log_probs)
                    all_values.append(value)

            # Convert to numpy arrays
            # actions shape: (n_envs, n_agents, action_dim)
            actions_array = np.array(all_actions)
            log_probs_array = np.array(all_log_probs)
            values_array = np.array(all_values)

            # IMPORTANT: Reshape actions for discrete environments
            if self.discrete:
                # For discrete actions, squeeze the last dimension
                # Shape: (n_envs, n_agents, 1) -> (n_envs, n_agents)
                if actions_array.ndim == 3 and actions_array.shape[-1] == 1:
                    actions_array = actions_array.squeeze(-1)

                # Ensure integer type
                actions_array = actions_array.astype(np.int32)

            # Step all environments in parallel
            # Gymnasium VectorEnv step returns:
            # - next_obs: (n_envs, n_agents, obs_dim)
            # - rewards: (n_envs,)
            # - terminateds: (n_envs,)
            # - truncateds: (n_envs,)
            # - infos: dict with keys that have (n_envs,) shape
            next_obs, rewards, terminateds, truncateds, infos = self.vec_env.step(
                actions_array
            )

            # Compute dones for each environment
            dones = np.logical_or(terminateds, truncateds)

            # Store transitions for all environments
            for env_idx in range(batch_size):
                # Extract individual rewards for each agent from info
                # Your SalpChainEnv returns info['local_rewards']
                if "local_rewards" in infos:
                    # infos['local_rewards'] has shape (n_envs, n_agents)
                    individual_rewards = infos["local_rewards"][env_idx]

                # For storage, we need to restore the action dimension if discrete
                # because the buffer expects (n_agents, 1) for discrete
                actions_to_store = actions_array[env_idx]
                if self.discrete and actions_to_store.ndim == 1:
                    actions_to_store = actions_to_store.reshape(-1, 1)

                self.agent.store_transition(
                    env_idx,
                    obs[env_idx],
                    global_states[env_idx],
                    actions_to_store,
                    individual_rewards + rewards[env_idx],
                    log_probs_array[env_idx],
                    values_array[env_idx],
                    np.array([dones[env_idx]] * self.n_agents),
                )

            # Update for next iteration
            obs = next_obs
            total_step_count += batch_size
            current_episode_steps += 1

            # Check for episode terminations
            for env_idx in range(batch_size):
                if dones[env_idx]:
                    steps_per_episode.append(current_episode_steps[env_idx])
                    current_episode_steps[env_idx] = 0
                    episode_count += 1

            # Gymnasium VectorEnv automatically resets terminated environments
            # The obs returned is already the reset observation for terminated envs

        # Get final values for advantage computation
        final_global_states = obs.reshape(batch_size, -1)
        final_values = []

        with torch.no_grad():
            for env_idx in range(batch_size):
                global_state_tensor = (
                    torch.FloatTensor(final_global_states[env_idx])
                    .unsqueeze(0)
                    .to(self.device)
                )
                value = (
                    self.agent.network_old.get_value(global_state_tensor).cpu().item()
                )
                final_values.append(value)

        return total_step_count, episode_count, steps_per_episode, final_values

    def train(self, total_steps, batch_size, minibatches, epochs, log_every=10000):
        """Train MAPPO agent"""
        print(f"Starting MAPPO training for {total_steps} total environment steps...")

        steps_completed = 0
        episodes_completed = 0

        self.training_stats["total_steps"] = []
        self.training_stats["reward"] = []
        self.training_stats["episodes"] = []
        self.training_stats["steps_per_episode"] = []

        while steps_completed < total_steps:
            steps_to_collect = min(batch_size, total_steps - steps_completed)

            # Collect trajectory
            step_count, episode_count, steps_per_episode, final_values = (
                self.collect_trajectory(max_steps=int(steps_to_collect))
            )

            # Update agent
            stats = self.agent.update(
                next_value=final_values,
                minibatch_size=batch_size // minibatches,
                epochs=epochs,
            )

            # Update tracking
            steps_completed += step_count
            episodes_completed += episode_count

            # Store statistics
            for key, value in stats.items():
                self.training_stats[key].append(value)

            # Evaluate
            rew_per_episode = []
            eval_episodes = 10
            while len(rew_per_episode) < eval_episodes:
                rew_per_episode.append(self.evaluate())
            eval_rewards = np.array(rew_per_episode).mean()

            self.training_stats["total_steps"].append(steps_completed)
            self.training_stats["reward"].append(eval_rewards)
            self.training_stats["episodes"].append(episodes_completed)
            self.training_stats["steps_per_episode"].extend(steps_per_episode)

            # Log progress
            if steps_completed % log_every < step_count:
                print(
                    f"Steps: {steps_completed}/{total_steps} ({steps_completed/total_steps*100:.1f}%) | "
                    f"Episodes: {episodes_completed} | "
                    f"Recent Avg Reward: {self.training_stats['reward'][-1]:.2f} | "
                    f"Last Batch Steps: {step_count}"
                )

                self.save_training_stats(
                    self.dirs["logs"] / "training_stats_checkpoint.pkl"
                )
                self.save_agent(self.dirs["models"] / "models_checkpoint.pth")

        print(
            f"Training completed! Total steps: {steps_completed}, Episodes: {episodes_completed}"
        )

    def evaluate(self, render=False):
        """Evaluate current policy"""

        # Set policies to eval
        self.agent.network_old.eval()

        with torch.no_grad():

            obs = self.wrapped_env.reset()

            episode_rew = 0

            while True:

                global_state = np.concatenate(obs)

                actions, _, _ = self.agent.get_actions(
                    obs, global_state, deterministic=True
                )

                obs, global_reward, local_rewards, terminated, truncated, info = (
                    self.wrapped_env.step(actions)
                )

                episode_rew += local_rewards[0] + global_reward

                if render:
                    self.wrapped_env.env.render()

                if terminated or truncated:
                    break

        # Set policies to train
        self.agent.network_old.train()

        return episode_rew

    def save_agent(self, path):
        """Save MAPPO agent"""
        torch.save(
            {
                "network": self.agent.network_old.state_dict(),
                "optimizer": self.agent.optimizer.state_dict(),
            },
            path,
        )

    def load_agent(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)

        self.agent.network_old.load_state_dict(checkpoint["network"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"Agents loaded from {filepath}")

    def save_training_stats(self, path):
        """Save training statistics"""
        with open(path, "wb") as f:
            pickle.dump(dict(self.training_stats), f)
