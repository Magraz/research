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

        # Create independent PPO agents
        state_dim = self.env.observation_space.shape[1]  # 17 features per agent
        action_dim = self.env.action_space.shape[1]  # 2 actions per agent

        self.agents = []
        for i in range(self.env.n_agents):
            agent = PPOAgent(
                state_dim=state_dim, action_dim=action_dim, device=device, **ppo_config
            )
            self.agents.append(agent)

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_stats = defaultdict(list)

    def collect_trajectory(self, max_steps=1000):
        obs, _ = self.env.reset()
        episode_reward = 0
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

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(
                np.array(actions)
            )

            # Store transitions for all agents
            for i, agent in enumerate(self.agents):
                agent.store_transition(
                    state=obs[i],
                    action=actions[i],
                    reward=reward,  # Global reward shared by all agents
                    log_prob=log_probs[i],
                    value=values[i],
                    done=terminated or truncated,
                )

            obs = next_obs
            episode_reward += reward
            step_count += 1

            if terminated or truncated:
                break

        # Get final values for advantage computation
        final_values = []
        for i, agent in enumerate(self.agents):
            _, _, final_value = agent.get_action(obs[i])
            final_values.append(final_value)

        return episode_reward, step_count, final_values

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

    def train(self, num_episodes=1000, log_every=10):
        print(f"Starting training for {num_episodes} episodes...")
        print(f"State dimension: {self.env.observation_space.shape[1]}")
        print(f"Action dimension: {self.env.action_space.shape[1]}")
        print(f"Number of agents: {self.env.n_agents}")

        for episode in range(num_episodes):
            episode_reward, episode_length = self.train_episode()

            # Logging
            if episode % log_every == 0:
                avg_reward = np.mean(self.episode_rewards[-log_every:])
                avg_length = np.mean(self.episode_lengths[-log_every:])

                print(
                    f"Episode {episode:4d} | "
                    f"Avg Reward: {avg_reward:8.2f} | "
                    f"Avg Length: {avg_length:6.1f} | "
                )

        print("Training completed!")

    def render_episode(self, max_steps=1000):
        obs, _ = self.env.reset()

        for step in range(max_steps):
            # Get actions from all agents (deterministic)
            actions = []
            for i, agent in enumerate(self.agents):
                with torch.no_grad():
                    state_tensor = (
                        torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
                    )
                    action_mean, _, _ = agent.network.forward(state_tensor)
                    action = torch.tanh(action_mean).cpu().numpy()[0]
                    actions.append(action)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(np.array(actions))
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
