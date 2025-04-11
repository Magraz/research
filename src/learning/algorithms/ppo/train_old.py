import os
import torch


from learning.environments.types import EnvironmentParams
from learning.environments.create_env import create_env
from learning.algorithms.ippo.ippo import PPO
from learning.algorithms.ppo.types import Experiment, Params

import numpy as np
import pickle as pkl
from pathlib import Path
from dataclasses import make_dataclass
import time
from statistics import mean

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
        n_envs: int,
        state: list,
        representation: str,
    ):
        match (representation):
            case "global":
                return state[0]
            case _:
                state = torch.stack(state).permute(1, 0, 2).reshape(n_envs, -1)
                return state

        return state

    def train(
        self,
        exp_config: Experiment,
        env_config: EnvironmentParams,
    ):

        # Set params
        params = Params(**exp_config.params)

        env = create_env(
            self.batch_dir,
            env_config.n_envs,
            device=self.device,
            env_name=env_config.environment,
            seed=params.random_seed,
        )

        # NumPy
        np.random.seed(params.random_seed)

        # PyTorch (CPU)
        torch.manual_seed(params.random_seed)
        torch.cuda.manual_seed(params.random_seed)

        params.device = self.device
        params.log_filename = self.logs_dir
        params.n_agents = env_config.n_agents
        params.action_dim = env.action_space.spaces[0].shape[0]
        params.state_dim = env.observation_space.spaces[0].shape[0] * params.n_agents
        params.batch_size = env_config.n_envs * params.n_steps
        params.minibatch_size = params.batch_size // params.n_minibatches

        learner = PPO(params, n_envs=env_config.n_envs)
        total_steps = 0
        rmax = -1e5
        running_avg_reward = 0
        data = []
        iterations = 0

        # Start training loop
        start_time = time.time()

        while total_steps < params.n_total_steps:

            iterations += 1

            done = torch.zeros(env_config.n_envs, dtype=torch.bool)
            cum_rewards = torch.zeros(env_config.n_envs, dtype=torch.float32)
            rewards_per_episode = [[] for _ in range(env_config.n_envs)]

            state = env.reset()

            episode_data = []

            for _ in range(0, params.n_steps):
                total_steps += env_config.n_envs

                action = torch.clamp(
                    learner.select_action(
                        self.process_state(
                            env_config.n_envs, state, env_config.state_representation
                        )
                    ),
                    min=-1.0,
                    max=1.0,
                )

                action = action.reshape(
                    params.n_agents,
                    env_config.n_envs,
                    params.action_dim,
                )

                action_tensor_list = [agent for agent in action]

                state, reward, done, _ = env.step(action_tensor_list)

                # Store transition
                # episode_data.append((state, action, reward, done))

                learner.add_reward_terminal(reward[0], done)

                cum_rewards += reward[0]

                # Reset environments
                if torch.any(done):
                    indices = torch.nonzero(done, as_tuple=True)[0]
                    for idx in indices:
                        state = env.reset_at(index=idx.item())
                        rewards_per_episode[idx].append(cum_rewards[idx].item())
                        cum_rewards[idx] = 0

            means = [
                sum(sublist) / len(sublist)
                for sublist in rewards_per_episode
                if sublist
            ]

            if means == []:
                mean_rew = float(torch.mean(cum_rewards / params.n_steps))
            else:
                mean_rew = mean(means)

            # Append episode summary instead of per-step data
            data.append(mean_rew)

            print(
                f"Steps {total_steps}, Reward {"{:.2f}".format(mean_rew)}, Minutes {"{:.2f}".format((time.time() - start_time) / 60)}"
            )

            running_avg_reward = (
                0.99 * running_avg_reward + 0.01 * mean_rew
                if total_steps > 0
                else mean_rew
            )

            if running_avg_reward > rmax:
                print(
                    f"New best reward: {"{:.2f}".format(running_avg_reward)} at step {total_steps}"
                )
                rmax = running_avg_reward
                # learner.save(self.models_dir / "best_model")

            # Save checkpoint
            # if total_steps % 100000 == 0:
            #     learner.save(self.models_dir / f"checkpoint_{total_steps}")

            if iterations % 10 == 0:
                with open(self.models_dir / "data.dat", "wb") as f:
                    pkl.dump(data, f)
                    print(f"Saved model at step {total_steps}")
                    learner.save(self.models_dir / "best_model")

            learner.update()

    def view(self, exp_config: Experiment, env_config: EnvironmentParams):

        self.device = "cpu"

        # Set params
        params = Params(**exp_config.params)

        env = create_env(
            self.batch_dir,
            1,
            device=self.device,
            env_name=env_config.environment,
            seed=params.random_seed,
        )

        params.device = self.device
        params.action_dim = env.action_space.spaces[0].shape[0]
        params.state_dim = (
            env.observation_space.spaces[0].shape[0] * env_config.n_agents
        )
        params.n_agents = env_config.n_agents
        params.log_data = False

        learner = PPO(params)
        learner.load(self.models_dir / "best_model")

        frame_list = []

        n_rollouts = 3

        for i in range(n_rollouts):
            done = False
            state = env.reset()
            R = torch.zeros(1, device=self.device)

            r = []
            while not done:

                action = torch.clamp(
                    learner.deterministic_action(
                        self.process_state(1, state, env_config.state_representation)
                    ),
                    min=-1.0,
                    max=1.0,
                )
                action = action.reshape(
                    env_config.n_agents,
                    1,
                    params.action_dim,
                )
                action_tensor_list = [agent for agent in action]

                state, reward, done, _ = env.step(action_tensor_list)

                r.append(reward)
                R += reward[0]

                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=True,
                )

                frame_list.append(frame)

            print(f"TOTAL RETURN: {R}")
            print(f"MAX {max(r)}")
            print(f"MIN {min(r)}")

        save_video(self.video_name, frame_list, fps=1 / env.scenario.world.dt)
