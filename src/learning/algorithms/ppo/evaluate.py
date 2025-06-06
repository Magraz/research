import os
import torch
import numpy as np

from learning.environments.types import EnvironmentParams
from learning.environments.create_env import create_env
from learning.algorithms.ppo.types import Experiment, Params
from learning.algorithms.ppo.ppo import PPO
from learning.algorithms.ppo.utils import get_state_dim, process_state
from learning.plotting.utils import (
    plot_attention_heatmap,
    plot_attention_time_series,
    plot_attention_over_time_grid,
    plot_key_attention_trends,
    plot_token_attention_trends,
)

import dill
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

from vmas.simulator.utils import save_video

matplotlib.use("Agg")


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
        self.plots_dir = self.trial_dir / "plots"
        self.models_dir = self.trial_dir / "models"
        self.video_dir = self.trial_dir / "video"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        for n_agents in range(4, 65, 4):
            (self.plots_dir / str(n_agents)).mkdir(parents=True, exist_ok=True)

    def validate(
        self,
        exp_config: Experiment,
        env_config: EnvironmentParams,
    ):

        params = Params(**exp_config.params)

        # Create environment to get dimension data
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

        # Get attention plots
        self.get_attention_plots(
            env,
            params,
            env_config.n_agents,
            d_state,
            d_action,
            exp_config,
            env_config,
        )

        # Get scalability plots
        self.get_scalability_plots(
            env,
            params,
            env_config.n_agents,
            d_state,
            d_action,
            exp_config,
            env_config,
            n_rollouts=50,
            extra_agents=64,
        )

    def get_scalability_plots(
        self,
        env,
        params: Params,
        n_agents: int,
        d_state: int,
        d_action: int,
        exp_config: Experiment,
        env_config: EnvironmentParams,
        n_rollouts: int,
        extra_agents: int,
    ):
        n_agents_list = list(range(4, extra_agents + 1, 4))
        seed = 1998
        data = {n_agents: [] for n_agents in n_agents_list}

        for i, n_agents in enumerate(n_agents_list):

            # Load environment and policy
            env = create_env(
                self.batch_dir,
                n_rollouts,
                device=self.device,
                env_name=env_config.environment,
                training=False,
                n_agents=n_agents,
                seed=seed,
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

            # Set policy to evaluation mode
            learner.policy.eval()

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

                cumulative_rewards = reward[0]

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

        # Store environment
        with open(self.logs_dir / "evaluation.dat", "wb") as f:
            dill.dump(data, f)

    def get_attention_plots(
        self,
        env,
        params: Params,
        n_agents: int,
        d_state: int,
        d_action: int,
        exp_config: Experiment,
        env_config: EnvironmentParams,
        extra_agents: int = 64,
    ):

        n_agents_list = list(range(8, extra_agents + 1, 16))
        seed = 1998
        attention_dict = {}

        for i, n_agents in enumerate(n_agents_list):

            # Load environment
            env = create_env(
                self.batch_dir,
                1,
                device=self.device,
                env_name=env_config.environment,
                training=False,
                n_agents=n_agents,
                seed=seed,
            )

            # Load PPO agent
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

            # Set policy to evaluation mode
            learner.policy.eval()

            edge_indices = []
            attention_weights = []
            attention_over_time = {
                "Enc_L0": [],  # Encoder self-attention
                "Dec_L0": [],  # Decoder self-attention
                "Cross_L0": [],  # Cross-attention
            }
            match (exp_config.model):
                case (
                    "transformer"
                    | "transformer_full"
                    | "transformer_encoder"
                    | "transformer_decoder"
                ):
                    attention_weights = learner.policy.build_attention_hooks()

            # Frame list for vide
            frames = []

            # Reset environment
            state = env.reset()

            for _ in range(0, params.n_max_steps_per_episode):

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

                match (exp_config.model):
                    case "gat" | "graph_transformer":
                        x = learner.policy.get_batched_graph(
                            process_state(
                                state,
                                env_config.state_representation,
                                exp_config.model,
                            )
                        )
                        _, attention_layers = learner.policy.forward_evaluation(x)

                        # Store edge indices and weights from last layer
                        # Make sure to do a deep copy
                        edge_index, attn_weight = attention_layers[-1]

                        # Store completely detached copies
                        edge_indices.append(edge_index.clone())
                        attention_weights.append(attn_weight.clone())

                    case (
                        "transformer"
                        | "transformer_full"
                        | "transformer_encoder"
                        | "transformer_decoder"
                    ):
                        _ = learner.policy.forward(
                            process_state(
                                state,
                                env_config.state_representation,
                                exp_config.model,
                            )
                        )

                        # Store attention weights for this timestep
                        for attn_type in attention_over_time:
                            if attn_type in attention_weights:
                                attention_over_time[attn_type].append(
                                    attention_weights[attn_type].clone()
                                )

                action_tensor = action.reshape(
                    1,
                    n_agents,
                    d_action,
                ).transpose(1, 0)

                # Turn action tensor into list of tensors with shape (n_env, action_dim)
                action_tensor_list = torch.unbind(action_tensor)

                state, _, done, _ = env.step(action_tensor_list)

                # Store frames for video
                frames.append(
                    env.render(
                        mode="rgb_array",
                        agent_index_focus=None,  # Can give the camera an agent index to focus on
                        visualize_when_rgb=False,
                    )
                )

                if torch.any(done):
                    break

            # Save video
            save_video(
                str(self.video_dir / f"plots_video_{n_agents}"),
                frames,
                fps=1 / env.scenario.world.dt,
            )

            # Store environment
            attention_dict[n_agents] = {
                "edge_indices": edge_indices,
                "attention_weights": attention_weights,
                "attention_over_time": attention_over_time,
            }

        with open(self.logs_dir / "attention.dat", "wb") as f:
            dill.dump(attention_dict, f)
