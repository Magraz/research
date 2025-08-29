import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from learning.environments.box2d_salp.domain import SalpChainEnv
from learning.algorithms.ippo.ippo import PPOAgent
import pickle  # Add this import at the top of the file


class IPPOTrainer:
    def __init__(self, env_config, ppo_config, device="cpu"):
        self.device = device
        self.env_config = env_config
        self.ppo_config = ppo_config

        # Create environment
        self.env = SalpChainEnv(**env_config)

        # Check if we're dealing with Dict action space
        self.using_dict_action = hasattr(self.env.action_space, "spaces")

        # Create independent PPO agents
        state_dim = self.env.observation_space.shape[1]

        if self.using_dict_action:
            # For Dict actions, pass the entire action_space to the agents
            self.agents = []
            for i in range(self.env.n_agents):
                agent = PPOAgent(
                    state_dim=state_dim,
                    action_space=self.env.action_space,  # Pass the whole action space
                    device=device,
                    **ppo_config,
                )
                self.agents.append(agent)
        else:
            # Original initialization for Box action space
            action_dim = self.env.action_space.shape[1]
            self.agents = []
            for i in range(self.env.n_agents):
                agent = PPOAgent(
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

    def collect_trajectory(self, max_steps=1000):
        obs, _ = self.env.reset()
        total_reward = 0
        step_count = 0

        for step in range(max_steps):
            # Get actions from all agents
            actions = []
            log_probs = []
            values = []

            for i, agent in enumerate(self.agents):
                action, log_prob, value = agent.get_action(obs[i])
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)

            # Format actions correctly for environment step
            if self.using_dict_action:
                # Convert list of dict actions to dict of batched actions
                env_actions = {}
                for key in actions[0].keys():
                    env_actions[key] = np.array([a[key] for a in actions])
            else:
                # Original format for Box actions
                env_actions = np.array(actions)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(env_actions)

            # Store transitions for all agents
            for i, agent in enumerate(self.agents):
                agent.store_transition(
                    state=obs[i],
                    action=actions[i],
                    reward=reward,
                    log_prob=log_probs[i],
                    value=values[i],
                    done=terminated or truncated,
                )

            obs = next_obs
            total_reward += reward
            step_count += 1

            # If environment terminated or truncated, reset it and continue collecting
            if terminated or truncated:
                # Get new observations from reset
                obs, _ = self.env.reset()

                # Note: We've already stored the transition with done=True
                # So the agents will know where episodes end for return calculation

        # Get final values for advantage computation
        final_values = []
        for i, agent in enumerate(self.agents):
            _, _, final_value = agent.get_action(obs[i])
            final_values.append(final_value)

        return total_reward, step_count, final_values

    def train_episode(self):
        # Collect trajectory
        episode_reward, episode_length, final_values = self.collect_trajectory()

        # Update all agents
        update_stats = {}
        for i, (agent, final_value) in enumerate(zip(self.agents, final_values)):
            stats = agent.update(next_value=final_value)
            for key, value in stats.items():
                if f"agent_{i}_{key}" not in update_stats:
                    update_stats[f"agent_{i}_{key}"] = []
                update_stats[f"agent_{i}_{key}"].append(value)

        # Store episode statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)

        # Store training statistics
        for key, values in update_stats.items():
            self.training_stats[key].extend(values)

        return episode_reward, episode_length

    def train(self, total_steps=1000000, log_every=1000):
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

        # Set a reasonable trajectory collection length
        collection_size = 1000

        # Keep training until we reach the desired number of steps
        while steps_completed < total_steps:
            # Determine how many more steps to collect in this iteration
            steps_to_collect = min(collection_size, total_steps - steps_completed)

            # Collect trajectory for a fixed number of steps
            episode_reward, step_count, final_values = self.collect_trajectory(
                max_steps=steps_to_collect
            )

            # Update all agents
            update_stats = {}
            for i, (agent, final_value) in enumerate(zip(self.agents, final_values)):
                stats = agent.update(next_value=final_value)
                for key, value in stats.items():
                    if f"agent_{i}_{key}" not in update_stats:
                        update_stats[f"agent_{i}_{key}"] = []
                    update_stats[f"agent_{i}_{key}"].append(value)

            # Update tracking variables
            steps_completed += step_count
            episodes_completed += 1

            # Store training statistics
            for key, values in update_stats.items():
                self.training_stats[key].extend(values)

            self.training_stats["total_steps"].append(steps_completed)
            self.training_stats["reward"].append(episode_reward)
            self.training_stats["episodes"].append(episodes_completed)

            # Store episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step_count)

            # Log progress
            if steps_completed % log_every < step_count:
                # Average over recent episodes
                recent_rewards = (
                    self.episode_rewards[-10:]
                    if len(self.episode_rewards) > 10
                    else self.episode_rewards
                )
                avg_reward = sum(recent_rewards) / len(recent_rewards)

                print(
                    f"Steps: {steps_completed}/{total_steps} ({steps_completed/total_steps*100:.1f}%) | "
                    f"Episodes: {episodes_completed} | "
                    f"Recent Avg Reward: {avg_reward:.2f} | "
                    f"Last Batch Steps: {step_count}"
                )

        print(
            f"Training completed! Total steps: {steps_completed}, Episodes: {episodes_completed}"
        )

    def render_episode(self, max_steps=1000):
        obs, _ = self.env.reset()

        for step in range(max_steps):
            # Get actions from all agents (deterministic)
            actions = []
            for i, agent in enumerate(self.agents):
                with torch.no_grad():
                    # Get deterministic action from the policy
                    state_tensor = (
                        torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
                    )
                    if self.using_dict_action:
                        # For Dict actions, use the network's get_action method
                        action, _, _ = agent.get_action(obs[i], deterministic=True)
                    else:
                        # Original approach for Box actions
                        action_mean, _, _ = agent.network.forward(state_tensor)
                        action = torch.tanh(action_mean).cpu().numpy()[0]

                    actions.append(action)

            # Format actions correctly for environment step
            if self.using_dict_action:
                env_actions = {}
                for key in actions[0].keys():
                    env_actions[key] = np.array([a[key] for a in actions])
            else:
                env_actions = np.array(actions)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(env_actions)
            self.env.render()

            if terminated or truncated:
                break

    def save_agents(self, filepath):
        torch.save(
            {
                "agents": [agent.network.state_dict() for agent in self.agents],
                "config": self.ppo_config,
                "env_config": self.env_config,
            },
            filepath,
        )
        print(f"Agents saved to {filepath}")

    def load_agents(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)

        for i, agent in enumerate(self.agents):
            agent.network.load_state_dict(checkpoint["agents"][i])

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
