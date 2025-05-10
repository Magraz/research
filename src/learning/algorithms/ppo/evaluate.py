import os
import torch
import numpy as np

from learning.environments.types import EnvironmentParams
from learning.environments.create_env import create_env
from learning.algorithms.ppo.types import Experiment, Params
from learning.algorithms.ppo.ppo import PPO

import dill
from pathlib import Path
import matplotlib.pyplot as plt


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

    def process_state(
        self,
        state: list,
        representation: str,
        model: str,
    ):
        match (model):
            case "mlp":
                match (representation):
                    case "global":
                        return state[0]

                    case "local":
                        state = torch.stack(state).transpose(1, 0).flatten(start_dim=1)

                        return state

            case "transformer" | "gat":
                match (representation):
                    case "local":
                        state = torch.stack(state).transpose(1, 0)
                        return state

        return state

    def get_state_dim(
        self, obs_shape, state_representation: str, model: str, n_agents: int
    ):

        match (model):
            case "mlp":
                match (state_representation):
                    case "global":
                        return obs_shape

                    case "local":
                        return obs_shape * n_agents

            case "transformer" | "gat":
                match (state_representation):
                    case "local":
                        return obs_shape

    def validate(
        self,
        exp_config: Experiment,
        env_config: EnvironmentParams,
        n_rollouts: int = 10,
        extra_agents: int = 4,
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
        d_state = self.get_state_dim(
            env.observation_space.spaces[0].shape[0],
            env_config.state_representation,
            exp_config.model,
            env_config.n_agents,
        )

        n_agents_list = [env_config.n_agents + i for i in range(extra_agents)]
        data = {n_agents: [] for n_agents in n_agents_list}

        for n_agents in n_agents_list:

            env = create_env(
                self.batch_dir,
                1,
                device=self.device,
                env_name=env_config.environment,
                n_agents=n_agents,
                seed=params.random_seed + n_agents,
            )
            learner = PPO(
                self.device,
                exp_config.model,
                params,
                env_config.n_agents,
                n_agents,
                1,
                d_state,
                d_action,
            )
            learner.load(self.models_dir / "best_model")

            for i in range(n_rollouts):

                state = env.reset()
                cumulative_reward = 0
                rewards = []

                for _ in range(0, params.n_max_steps_per_episode):

                    action = torch.clamp(
                        learner.deterministic_action(
                            self.process_state(
                                state,
                                env_config.state_representation,
                                exp_config.model,
                            )
                        ),
                        min=-1.0,
                        max=1.0,
                    )

                    action_tensor = action.reshape(
                        1,
                        n_agents,
                        d_action,
                    ).transpose(1, 0)

                    # Turn action tensor into list of tensors with shape (n_env, action_dim)
                    action_tensor_list = torch.unbind(action_tensor)

                    state, reward, done, _ = env.step(action_tensor_list)

                    cumulative_reward += reward[0]

                    if torch.any(done):
                        print("DONE")
                        break

                print(f"n_agents: {n_agents}, reward: {cumulative_reward}")
                rewards.append(cumulative_reward)

            mean_rew = torch.stack(rewards).mean().item()
            data[n_agents].append(mean_rew)

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
            plt.fill_between(
                n_agents,
                [r - e for r, e in zip(rewards, errors)],
                [r + e for r, e in zip(rewards, errors)],
                alpha=0.2,
            )

        # Customize plot
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.title("Reward vs. Number of Agents", fontsize=16)
        plt.xlabel("Number of Agents", fontsize=14)
        plt.ylabel("Mean Reward", fontsize=14)
        plt.xticks(n_agents)  # Ensure x-axis shows exact agent numbers

        # Add value labels above each point
        # for i, reward in enumerate(rewards):
        #     plt.annotate(
        #         f"{reward:.2f}",
        #         (n_agents[i], rewards[i]),
        #         textcoords="offset points",
        #         xytext=(0, 10),
        #         ha="center",
        #     )

        plt.tight_layout()

        # Save if requested
        plt.savefig(
            Path(self.logs_dir) / "agents_vs_reward.png", dpi=300, bbox_inches="tight"
        )

        # Show if requested
        plt.show()
        plt.close()
