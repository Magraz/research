import os
import torch
import numpy as np

from learning.environments.types import EnvironmentParams
from learning.environments.create_env import create_env
from learning.algorithms.ppo.types import Experiment, Params
from learning.algorithms.ppo.ppo import PPO
from learning.algorithms.ppo.utils import get_state_dim, process_state

import dill
from pathlib import Path
import matplotlib.pyplot as plt
import random
from statistics import mean


class PPO_Evaluator:
    def __init__(
        self,
        device: str,
        batch_dir: Path,
        trials_dir: Path,
        trial_id: str,
        video_name: str,
    ):
        # Directories
        self.device = device
        self.batch_dir = batch_dir
        self.video_name = video_name
        self.trial_dir = trials_dir / trial_id
        self.logs_dir = self.trial_dir / "logs"
        self.models_dir = self.trial_dir / "models"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def validate(
        self,
        exp_config: Experiment,
        env_config: EnvironmentParams,
        n_rollouts: int = 50,
        extra_agents: int = 16,
    ):

        params = Params(**exp_config.params)

        env = create_env(
            self.batch_dir,
            1,
            device=self.device,
            env_name=env_config.environment,
            n_agents=env_config.n_agents,
            seed=params.random_seed,
        )

        d_action = env.action_space.spaces[0].shape[0]
        d_state = get_state_dim(
            env.observation_space.spaces[0].shape[0],
            env_config.state_representation,
            exp_config.model,
            env_config.n_agents,
        )

        n_agents_list = [env_config.n_agents + i for i in range(extra_agents)]
        data = {n_agents: [] for n_agents in n_agents_list}

        for n_agents in n_agents_list:

            # Load environment and policy
            env = create_env(
                self.batch_dir,
                n_rollouts,
                device=self.device,
                env_name=env_config.environment,
                training=False,
                n_agents=n_agents,
                seed=params.random_seed + random.randint(1, 100),
            )
            learner = PPO(
                self.device,
                exp_config.model,
                params,
                env_config.n_agents,
                n_agents,
                n_rollouts,
                d_state,
                d_action,
            )
            learner.load(self.models_dir / "best_model")

            rewards = []
            episode_count = 0
            state = env.reset()
            cumulative_rewards = torch.zeros(
                n_rollouts, dtype=torch.float32, device=self.device
            )
            episode_len = torch.zeros(n_rollouts, dtype=torch.int32, device=self.device)

            for step in range(0, params.n_max_steps_per_episode):

                action = torch.clamp(
                    learner.deterministic_action(
                        process_state(
                            state,
                            env_config.state_representation,
                            exp_config.model,
                        )
                    ),
                    min=-1.0,
                    max=1.0,
                )

                action_tensor = action.reshape(
                    n_rollouts,
                    n_agents,
                    d_action,
                ).transpose(1, 0)

                # Turn action tensor into list of tensors with shape (n_env, action_dim)
                action_tensor_list = torch.unbind(action_tensor)

                state, reward, done, _ = env.step(action_tensor_list)

                cumulative_rewards += reward[0]

                episode_len += torch.ones(
                    n_rollouts, dtype=torch.int32, device=self.device
                )

                # Create timeout boolean mask
                timeout = episode_len == params.n_max_steps_per_episode

                if torch.any(done) or torch.any(timeout):

                    # Get done and timeout indices
                    done_indices = torch.nonzero(done).flatten().tolist()
                    timeout_indices = torch.nonzero(timeout).flatten().tolist()

                    # Merge indices and remove duplicates
                    indices = list(set(done_indices + timeout_indices))

                    for idx in indices:
                        # Log data when episode is done
                        rewards.append(cumulative_rewards[idx].item())

                        # Reset vars, and increase counters
                        state = env.reset_at(index=idx)
                        cumulative_rewards[idx] = 0

                        episode_count += 1

                if episode_count >= n_rollouts:
                    break

            data[n_agents] = rewards

            print(data)

        # Store environment
        with open(self.logs_dir / "evaluation.dat", "wb") as f:
            dill.dump(data, f)

        # Plot
        n_agents = list(data.keys())
        rewards = [np.mean(data[n]) for n in n_agents]

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(n_agents, rewards, "o-", linewidth=2, markersize=8)

        # Add error bars if there are multiple rewards per agent count
        errors = [np.std(data[n]) if len(data[n]) > 1 else 0 for n in n_agents]
        if any(errors):
            # plt.fill_between(
            #     n_agents,
            #     [r - e for r, e in zip(rewards, errors)],
            #     [r + e for r, e in zip(rewards, errors)],
            #     alpha=0.2,
            # )

            plt.errorbar(
                n_agents,
                rewards,
                yerr=errors,
                fmt="o-",  # This maintains your original line style
                linewidth=2,
                markersize=8,
                capsize=5,  # Adds caps to the error bars
                ecolor="gray",  # Optional: set error bar color
            )

        # Customize plot
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.title("Reward vs. Number of Agents", fontsize=16)
        plt.xlabel("Number of Agents", fontsize=14)
        plt.ylabel("Mean Reward", fontsize=14)
        plt.xticks(n_agents)  # Ensure x-axis shows exact agent numbers

        plt.tight_layout()

        # Save if requested
        plt.savefig(
            Path(self.logs_dir) / "agents_vs_reward.png", dpi=300, bbox_inches="tight"
        )
