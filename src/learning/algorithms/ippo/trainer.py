import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from learning.environments.box2d_salp.domain import SalpChainEnv
from learning.algorithms.ippo.ippo import PPOAgent
import pickle  # Add this import at the top of the file

from learning.environments.types import EnvironmentEnum


class IPPOTrainer:
    def __init__(
        self,
        env,
        env_name,
        n_agents,
        state_dim,
        action_dim=None,
        ppo_config=None,
        dirs=None,
        device="cpu",
    ):
        self.device = device
        self.dirs = dirs
        self.ppo_config = ppo_config
        self.n_agents = n_agents

        # Create environment
        self.env = env
        self.env_name = env_name

        if self.env_name == EnvironmentEnum.MPE:
            self.action_low = 0
            self.action_high = 1
        else:
            self.action_low = -1
            self.action_high = 1

        self.agents = []

        for _ in range(self.n_agents):
            agent = PPOAgent(
                env_name,
                state_dim=state_dim,
                action_dim=action_dim,
                device=device,
                **ppo_config,
            )
            self.agents.append(agent)

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_stats = defaultdict(list)

    def collect_trajectory(self, max_steps):

        if self.env_name == EnvironmentEnum.MPE:
            obs, _ = self.env.reset()
            obs = np.stack(list(obs.values()))

        else:
            obs, _ = self.env.reset()

        total_reward = 0
        total_step_count = 0
        current_episode_steps = 0
        steps_per_episode = []
        episode_count = 0

        for step in range(max_steps):
            # Get actions from all agents
            actions = []
            log_probs = []
            values = []

            for i, agent in enumerate(self.agents):

                with torch.no_grad():
                    action, log_prob, value = agent.get_action(obs[i])
                    action = np.clip(action, self.action_low, self.action_high)

                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)

            env_actions = np.array(actions)

            # Step environment
            if self.env_name == EnvironmentEnum.MPE:
                action_dict = {}

                for i, agent_id in enumerate(self.env.agents):
                    action_dict[agent_id] = env_actions[i]

                next_obs, rewards, terminated, truncated, _ = self.env.step(action_dict)

                next_obs = np.stack(list(next_obs.values()))
                rewards = np.stack(list(rewards.values()))
                terminated = np.stack(list(terminated.values()))
                truncated = np.stack(list(truncated.values()))

            else:

                next_obs, shared_reward, terminated, truncated, info = self.env.step(
                    env_actions
                )
                rewards = np.array(info["individual_rewards"])

            # Store transitions for all agents
            for i, agent in enumerate(self.agents):
                agent.store_transition(
                    state=obs[i],
                    action=env_actions[i],
                    reward=rewards[i],
                    log_prob=log_probs[i],
                    value=values[i],
                    done=terminated[i] or truncated[i],
                )

            obs = next_obs
            total_reward += rewards[i]
            total_step_count += 1
            current_episode_steps += 1

            # If environment terminated or truncated, reset it and continue collecting
            if terminated.all() or truncated.all():
                # Get new observations from reset
                if self.env_name == EnvironmentEnum.MPE:
                    obs, _ = self.env.reset()
                    obs = np.stack(list(obs.values()))
                else:
                    obs, _ = self.env.reset()

                # Keep track of episode count
                steps_per_episode.append(current_episode_steps)
                current_episode_steps = 0
                episode_count += 1

                # Note: We've already stored the transition with done=True
                # So the agents will know where episodes end for return calculation

        # Get final values for advantage computation
        final_values = []
        for i, agent in enumerate(self.agents):
            with torch.no_grad():
                _, _, final_value = agent.get_action(obs[i])
            final_values.append(final_value)

        return (
            total_reward,
            total_step_count,
            episode_count,
            steps_per_episode,
            final_values,
        )

    def train(self, total_steps, batch_size, minibatches, log_every=1000):
        """
        Train agents for a specific number of environment steps.

        Args:
            total_steps: Total number of environment steps to train for
            log_every: Log progress every X steps
        """
        print(f"Starting training for {total_steps} total environment steps...")

        # Initialize tracking variables
        steps_completed = 0
        episodes_completed = 0
        self.training_stats["total_steps"] = []
        self.training_stats["reward"] = []
        self.training_stats["episodes"] = []
        self.training_stats["steps_per_episode"] = []

        # Keep training until we reach the desired number of steps
        while steps_completed < total_steps:
            # Determine how many more steps to collect in this iteration
            steps_to_collect = min(batch_size, total_steps - steps_completed)

            # Collect trajectory for a fixed number of steps
            (
                total_rewards,
                step_count,
                episode_count,
                steps_per_episode,
                final_values,
            ) = self.collect_trajectory(max_steps=int(steps_to_collect))

            # Update all agents
            update_stats = {}
            for i, (agent, final_value) in enumerate(zip(self.agents, final_values)):
                stats = agent.update(
                    next_value=final_value, minibatch_size=batch_size // minibatches
                )
                for key, value in stats.items():
                    if f"agent_{i}_{key}" not in update_stats:
                        update_stats[f"agent_{i}_{key}"] = []
                    update_stats[f"agent_{i}_{key}"].append(value)

            # Update tracking variables
            steps_completed += step_count
            episodes_completed += episode_count

            # Store training statistics like loss and entropy
            for key, values in update_stats.items():
                self.training_stats[key].extend(values)

            self.training_stats["total_steps"].append(steps_completed)
            self.training_stats["reward"].append(total_rewards / episode_count)
            self.training_stats["episodes"].append(episodes_completed)
            self.training_stats["steps_per_episode"].extend(steps_per_episode)

            # Log progress
            if steps_completed % log_every < step_count:
                # Average over recent batch updates
                # average_window = 100
                # recent_rewards = (
                #     self.training_stats["reward"][-average_window:]
                #     if len(self.training_stats["reward"]) > average_window
                #     else self.training_stats["reward"]
                # )
                # avg_reward = sum(recent_rewards) / len(recent_rewards)

                print(
                    f"Steps: {steps_completed}/{total_steps} ({steps_completed/total_steps*100:.1f}%) | "
                    f"Episodes: {episodes_completed} | "
                    f"Recent Avg Reward: {self.training_stats['reward'][-1]:.2f} | "
                    f"Last Batch Steps: {step_count}"
                )

                self.save_training_stats(
                    self.dirs["logs"] / "training_stats_checkpoint.pkl"
                )
                self.save_agents(self.dirs["models"] / "models_checkpoint.pth")

        print(
            f"Training completed! Total steps: {steps_completed}, Episodes: {episodes_completed}"
        )

    def render_episode(self, max_steps):

        if self.env_name == EnvironmentEnum.MPE:
            obs, _ = self.env.reset(seed=42)
            obs = np.stack(list(obs.values()))
        else:
            obs, _ = self.env.reset()

        cumulative_reward = 0

        for step in range(max_steps):
            # Get actions from all agents (deterministic)
            actions = []
            for i, agent in enumerate(self.agents):
                with torch.no_grad():
                    agent.policy_old.eval()
                    action, _, _ = agent.get_action(obs[i], deterministic=True)
                    action = np.clip(action, self.action_low, self.action_high)
                actions.append(action)

            # Format actions correctly for environment step
            env_actions = np.array(actions)

            # Step environment
            if self.env_name == EnvironmentEnum.MPE:
                next_obs = []
                rewards = []
                action_dict = {}

                for i, agent_id in enumerate(self.env.agents):
                    action_dict[agent_id] = env_actions[i]

                next_obs, rewards, terminated, truncated, _ = self.env.step(action_dict)

                next_obs = np.stack(list(next_obs.values()))
                rewards = np.stack(list(rewards.values()))
                terminated = np.stack(list(terminated.values()))
                truncated = np.stack(list(truncated.values()))

            else:
                next_obs, shared_reward, terminated, truncated, info = self.env.step(
                    env_actions
                )
                rewards = np.array(info["individual_rewards"])

            self.env.render()

            cumulative_reward += rewards[0]

            if (step + 1) >= max_steps:
                print(f"TIMEOUT REWARD: {cumulative_reward}")
                break

            if terminated.all() or truncated.all():
                print(f"REWARD: {cumulative_reward}")
                break

    def save_agents(self, filepath):
        torch.save(
            {
                "agents": [agent.policy_old.state_dict() for agent in self.agents],
            },
            filepath,
        )
        print(f"Agents saved to {filepath}")

    def load_agents(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)

        for i, agent in enumerate(self.agents):
            agent.policy.load_state_dict(checkpoint["agents"][i])
            agent.policy_old.load_state_dict(checkpoint["agents"][i])

        print(f"Agents loaded from {filepath}")

    def save_training_stats(self, filepath):
        """Save the training statistics to a file."""
        with open(filepath, "wb") as f:
            pickle.dump(self.training_stats, f)
        print(f"Training statistics saved to {filepath}")

    def load_training_stats(self, filepath):
        """Load the training statistics from a file."""
        with open(filepath, "rb") as f:
            self.training_stats = pickle.load(f)
        print(f"Training statistics loaded from {filepath}")
