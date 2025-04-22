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

    def process_state(
        self,
        state: list,
        representation: str,
    ):
        match (representation):
            case "global":
                return state[0]
            case _:
                # Need state to be in the shape (n_env, agent, state_dim)
                state = torch.stack(state).transpose(1, 0).flatten(start_dim=1)
                return state

        return state

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
        env = create_env(
            self.batch_dir,
            n_envs,
            device=self.device,
            env_name=env_config.environment,
            seed=params.random_seed,
        )

        params.device = self.device
        params.log_filename = self.logs_dir
        params.n_agents = env_config.n_agents
        params.action_dim = env.action_space.spaces[0].shape[0]

        # Check state representation
        match (env_config.state_representation):
            case "global":
                params.state_dim = env.observation_space.spaces[0].shape[0]
            case _:
                params.state_dim = (
                    env.observation_space.spaces[0].shape[0] * params.n_agents
                )

        # Create learner object
        learner = PPO(params=params, n_envs=n_envs)

        # Setup loop variables
        step = 0
        total_episodes = 0
        max_episodes_per_rollout = 10
        max_steps_per_episode = params.n_steps // max_episodes_per_rollout
        rmax = -1e6
        running_avg_reward = 0
        iterations = 0
        data = []

        # Log start time
        start_time = time.time()

        while step < params.n_total_steps:

            rollout_episodes = 0

            episode_len = torch.zeros(
                env_config.n_envs, dtype=torch.int32, device=params.device
            )
            cum_rewards = torch.zeros(
                env_config.n_envs, dtype=torch.float32, device=params.device
            )

            state = env.reset()

            for _ in range(0, params.n_steps):

                actions_per_env = torch.clamp(
                    learner.select_action(
                        self.process_state(
                            state,
                            env_config.state_representation,
                        )
                    ),
                    min=-1.0,
                    max=1.0,
                )

                # Permute action tensor of shape (n_envs, n_agents*action_dim) to (agents, n_env, action_dim)
                action_tensor = actions_per_env.reshape(
                    n_envs,
                    params.n_agents,
                    params.action_dim,
                ).transpose(1, 0)

                # Turn action tensor into list of tensors with shape (n_env, action_dim)
                action_tensor_list = torch.unbind(action_tensor)

                state, reward, done, _ = env.step(action_tensor_list)

                learner.add_reward_terminal(reward[0], done)

                cum_rewards += reward[0]

                step += env_config.n_envs
                episode_len += torch.ones(
                    env_config.n_envs, dtype=torch.int32, device=params.device
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

                if rollout_episodes == max_episodes_per_rollout:
                    break

            if running_avg_reward > rmax:
                print(f"New best reward: {running_avg_reward} at step {step}")
                rmax = running_avg_reward
                learner.save(self.models_dir / "best_model")

            # if step % 10000 == 0:
            #     learner.save(self.models_dir / f"checkpoint_{step}")

            with open(self.models_dir / "data.dat", "wb") as f:
                pkl.dump(data, f)

            learner.update()

            iterations += 1

    def view(self, exp_config: Experiment, env_config: EnvironmentParams):

        params = Params(**exp_config.params)

        env = create_env(
            self.batch_dir,
            1,
            device=self.device,
            env_name=env_config.environment,
            seed=params.random_seed,
        )

        params.device = self.device
        params.n_agents = env_config.n_agents
        params.action_dim = env.action_space.spaces[0].shape[0]
        # Check state representation
        match (env_config.state_representation):
            case "global":
                params.state_dim = env.observation_space.spaces[0].shape[0]
            case _:
                params.state_dim = (
                    env.observation_space.spaces[0].shape[0] * params.n_agents
                )

        learner = PPO(params=params)
        learner.load(self.models_dir / "best_model")

        frame_list = []

        n_rollouts = 3

        for i in range(n_rollouts):
            done = False
            state = env.reset()
            R = 0
            r = []
            for t in range(0, 512):

                action = torch.clamp(
                    learner.deterministic_action(
                        self.process_state(
                            state,
                            env_config.state_representation,
                        )
                    ),
                    min=-1.0,
                    max=1.0,
                )

                action_tensor = action.reshape(
                    1,
                    params.n_agents,
                    params.action_dim,
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
            # print(f"MAX {max(r)}")
            # print(f"MIN {min(r)}")

        save_video(self.video_name, frame_list, fps=1 / env.scenario.world.dt)
