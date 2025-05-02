import os
import torch
import numpy as np

from learning.environments.types import EnvironmentParams
from learning.environments.create_env import create_env
from learning.algorithms.ppo.types import Experiment, Params
from learning.algorithms.ppo.ppo import PPO

import pickle as pkl
from pathlib import Path
import random
import time

from vmas.simulator.utils import save_video


class PPO_Trainer:
    def __init__(
        self,
        device: str,
        batch_dir: Path,
        trials_dir: Path,
        trial_id: int,
        trial_name: str,
        video_name: str,
        checkpoint: bool,
    ):
        # Directories
        self.device = device
        self.batch_dir = batch_dir
        self.trials_dir = trials_dir
        self.trial_name = trial_name
        self.trial_id = trial_id
        self.video_name = video_name
        self.trial_folder_name = "_".join(("trial", str(self.trial_id)))
        self.trial_dir = self.trials_dir / self.trial_folder_name
        self.logs_dir = self.trial_dir / "logs"
        self.models_dir = self.trial_dir / "models"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint loading
        self.checkpoint = checkpoint

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
                        # Need state to be in the shape (n_env, agent, state_dim)
                        state = torch.stack(state).transpose(1, 0).flatten(start_dim=1)

                        return state

            case "transformer":
                match (representation):
                    case "local":
                        # Need state to be in the shape (n_env, agent, state_dim)
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

            case "transformer":
                match (state_representation):
                    case "local":
                        return obs_shape

    def train(
        self,
        exp_config: Experiment,
        env_config: EnvironmentParams,
    ):

        # Set params
        params = Params(**exp_config.params)

        # Set seeds
        np.random.seed(params.random_seed)
        random.seed(params.random_seed)
        torch.manual_seed(params.random_seed)
        torch.cuda.manual_seed(params.random_seed)

        n_envs = env_config.n_envs
        n_agents = env_config.n_agents

        env = create_env(
            self.batch_dir,
            n_envs,
            n_agents=n_agents,
            device=self.device,
            env_name=env_config.environment,
            seed=params.random_seed,
        )

        d_action = env.action_space.spaces[0].shape[0]
        d_state = self.get_state_dim(
            env.observation_space.spaces[0].shape[0],
            env_config.state_representation,
            exp_config.model,
            n_agents,
        )

        # Create learner object
        learner = PPO(
            self.device,
            exp_config.model,
            params,
            n_agents,
            n_envs,
            d_state,
            d_action,
        )

        # Checkpoint loading logic
        data = []
        if self.checkpoint:

            checkpoint_path = self.models_dir / "checkpoint"

            if checkpoint_path.is_file():
                # Load checkpoint
                learner.load(checkpoint_path)
                # Load data
                with open(self.models_dir / "data.dat", "rb") as f:
                    data = pkl.load(data)

        # Setup loop variables
        step = 0
        total_episodes = 0
        max_episodes_per_rollout = 10
        max_steps_per_episode = params.batch_size // max_episodes_per_rollout
        rmax = -1e6
        running_avg_reward = 0
        iterations = 0
        checkpoint_step = 0

        # Log start time
        start_time = time.time()

        while step < params.n_total_steps:

            rollout_episodes = 0

            episode_len = torch.zeros(
                env_config.n_envs, dtype=torch.int32, device=self.device
            )
            cum_rewards = torch.zeros(
                env_config.n_envs, dtype=torch.float32, device=self.device
            )

            state = env.reset()

            for _ in range(0, params.batch_size):

                # Clamp because actions are stochastic and can lead to them been out of -1 to 1 range
                actions_per_env = torch.clamp(
                    learner.select_action(
                        self.process_state(
                            state,
                            env_config.state_representation,
                            exp_config.model,
                        )
                    ),
                    min=-1.0,
                    max=1.0,
                )

                # Permute action tensor of shape (n_envs, n_agents*action_dim) to (agents, n_env, action_dim)
                action_tensor = actions_per_env.reshape(
                    n_envs,
                    n_agents,
                    d_action,
                ).transpose(1, 0)

                # Turn action tensor into list of tensors with shape (n_env, action_dim)
                action_tensor_list = torch.unbind(action_tensor)

                state, reward, done, _ = env.step(action_tensor_list)

                learner.add_reward_terminal(reward[0], done)

                cum_rewards += reward[0]

                step += env_config.n_envs
                episode_len += torch.ones(
                    env_config.n_envs, dtype=torch.int32, device=self.device
                )

                # Create timeout boolean mask
                timeout = episode_len == max_steps_per_episode

                if torch.any(done) or torch.any(timeout):

                    # Get done and timeout indices
                    done_indices = torch.nonzero(done).flatten().tolist()
                    timeout_indices = torch.nonzero(timeout).flatten().tolist()

                    # Merge indices and remove duplicates
                    indices = list(set(done_indices + timeout_indices))

                    for idx in indices:
                        # Log data when episode is done
                        r = cum_rewards[idx].item()

                        data.append(r)

                        print(
                            f"Step {step}, Reward: {r}, Minutes {"{:.2f}".format((time.time() - start_time) / 60)}"
                        )

                        running_avg_reward = (
                            0.99 * running_avg_reward + 0.01 * r
                            if total_episodes > 0
                            else r
                        )

                        # Reset vars, and increase counters
                        state = env.reset_at(index=idx)
                        cum_rewards[idx], episode_len[idx] = 0, 0

                        total_episodes += 1
                        rollout_episodes += 1

                if rollout_episodes >= max_episodes_per_rollout:
                    break

            if running_avg_reward > rmax:
                print(f"New best reward: {running_avg_reward} at step {step}")
                rmax = running_avg_reward
                learner.save(self.models_dir / "best_model")

            if step - checkpoint_step >= 10000:
                learner.save(self.models_dir / "checkpoint")
                checkpoint_step = step

            with open(self.models_dir / "data.dat", "wb") as f:
                pkl.dump(data, f)

            learner.update()

            iterations += 1

    def view(self, exp_config: Experiment, env_config: EnvironmentParams):

        params = Params(**exp_config.params)

        # Override training agent count
        # n_agents = 4
        # env_config.n_agents = n_agents

        n_agents = env_config.n_agents

        env = create_env(
            self.batch_dir,
            1,
            device=self.device,
            env_name=env_config.environment,
            n_agents=n_agents,
            seed=params.random_seed,
        )

        d_action = env.action_space.spaces[0].shape[0]
        d_state = self.get_state_dim(
            env.observation_space.spaces[0].shape[0],
            env_config.state_representation,
            exp_config.model,
            n_agents,
        )

        learner = PPO(
            self.device,
            exp_config.model,
            params,
            n_agents,
            1,
            d_state,
            d_action,
        )
        learner.load(self.models_dir / "best_model")

        frame_list = []

        n_rollouts = 5

        for i in range(n_rollouts):
            done = False
            state = env.reset()
            R = 0
            r = []
            for t in range(0, 512):

                action = learner.deterministic_action(
                    self.process_state(
                        state,
                        env_config.state_representation,
                        exp_config.model,
                    )
                )

                action_tensor = action.reshape(
                    1,
                    n_agents,
                    d_action,
                ).transpose(1, 0)

                # Turn action tensor into list of tensors with shape (n_env, action_dim)
                action_tensor_list = torch.unbind(action_tensor)

                state, reward, done, _ = env.step(action_tensor_list)

                r.append(reward)
                R += reward[0]

                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=True,
                )

                frame_list.append(frame)

                if torch.any(done):
                    print("DONE")
                    break

            print(f"TOTAL RETURN: {R}")

        save_video(self.video_name, frame_list, fps=1 / env.scenario.world.dt)
