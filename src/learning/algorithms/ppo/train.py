import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from learning.environments.types import EnvironmentParams
from learning.environments.create_env import create_env
from learning.algorithms.ppo.types import Experiment, Params
from learning.algorithms.ppo.ppo import PPO
from learning.algorithms.ppo.utils import get_state_dim, process_state

import pickle as pkl
import dill
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
        trial_id: str,
        video_name: str,
        checkpoint: bool,
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

        # Checkpoint loading
        self.checkpoint = checkpoint

    def train(
        self,
        exp_config: Experiment,
        env_config: EnvironmentParams,
    ):
        # Set logger
        self.writer = SummaryWriter(self.logs_dir)

        # Set params
        params = Params(**exp_config.params)

        # Set seeds
        np.random.seed(params.random_seed)
        random.seed(params.random_seed)
        torch.manual_seed(params.random_seed)
        torch.cuda.manual_seed(params.random_seed)

        # Create environment
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

        # Set state and action dimensions
        d_action = env.action_space.spaces[0].shape[0]
        d_state = get_state_dim(
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
            n_agents,
            n_envs,
            d_state,
            d_action,
            self.writer,
            self.checkpoint,
        )

        # Checkpoint loading logic
        data = []
        if self.checkpoint:

            checkpoint_path = self.models_dir / "checkpoint"

            if checkpoint_path.is_file():
                # Load checkpoint
                learner.load(checkpoint_path)

                # Load data
                with open(self.logs_dir / "data.dat", "rb") as data_file:
                    data = dill.load(data_file)

                # Load env up to checkpoint
                with open(self.models_dir / "env.dat", "rb") as env_file:
                    env = dill.load(env_file)

        # Setup loop variables
        global_step = 0
        checkpoint_step = 0
        total_episodes = 0
        rmax = -1e6
        running_avg_reward = 0

        # Log start time
        start_time = time.time()

        while global_step < params.n_total_steps:

            episode_len = torch.zeros(
                env_config.n_envs, dtype=torch.int32, device=self.device
            )
            cum_rewards = torch.zeros(
                env_config.n_envs, dtype=torch.float32, device=self.device
            )

            state = env.reset()

            # Collect batch of data stepping by n_envs
            for _ in range(0, params.batch_size, env_config.n_envs):

                # Clamp because actions are stochastic and can lead to them been out of -1 to 1 range
                actions_per_env = torch.clamp(
                    learner.select_action(
                        process_state(
                            state,
                            env_config.state_representation,
                            exp_config.model,
                        )
                    ),
                    min=-1.0,
                    max=1.0,
                )

                # Permute action tensor of shape (n_envs, n_agents * action_dim) to (agents, n_env, action_dim)
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

                episode_len += torch.ones(
                    env_config.n_envs, dtype=torch.int32, device=self.device
                )

                # Create timeout boolean mask
                timeout = episode_len == params.n_max_steps_per_episode

                # Increase counters
                global_step += env_config.n_envs

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
                            f"Step {global_step}, Reward: {r}, Minutes {"{:.2f}".format((time.time() - start_time) / 60)}"
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

            # Store best model if running average reward is higher than previous best
            if running_avg_reward > rmax:
                print(f"New best reward: {running_avg_reward} at step {global_step}")
                rmax = running_avg_reward
                learner.save(self.models_dir / "best_model")

            # Store checkpoint
            if global_step - checkpoint_step >= 10000:
                # Save model
                learner.save(self.models_dir / "checkpoint")

                # Store reward per episode data
                with open(self.logs_dir / "data.dat", "wb") as f:
                    dill.dump(data, f)

                # Store environment
                with open(self.models_dir / "env.dat", "wb") as f:
                    dill.dump(env, f)

                checkpoint_step = global_step

                print("CHECKPOINT SAVED")

            # Do training step
            learner.update()

            # Log reward data with tensorboard
            if self.writer is not None:
                for reward in data:
                    self.writer.add_scalar(
                        "Agent/rewards_per_episode", reward, total_episodes
                    )

        self.writer.close()

    def view(self, exp_config: Experiment, env_config: EnvironmentParams):

        params = Params(**exp_config.params)

        n_agents_train = env_config.n_agents
        n_agents_eval = 12
        n_rollouts = 3

        env = create_env(
            self.batch_dir,
            1,
            device=self.device,
            env_name=env_config.environment,
            n_agents=n_agents_eval,
            training=False,
            seed=params.random_seed + random.randint(1, 100),
        )

        d_action = env.action_space.spaces[0].shape[0]
        d_state = get_state_dim(
            env.observation_space.spaces[0].shape[0],
            env_config.state_representation,
            exp_config.model,
            n_agents_train,
        )

        learner = PPO(
            self.device,
            exp_config.model,
            params,
            n_agents_train,
            n_agents_eval,
            1,
            d_state,
            d_action,
        )
        learner.load(self.models_dir / "best_model")

        frame_list = []

        for i in range(n_rollouts):

            done = False
            state = env.reset()
            R = 0
            r = []

            for t in range(0, 512):

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
                    1,
                    n_agents_eval,
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
