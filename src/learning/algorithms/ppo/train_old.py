import os
import torch
import numpy as np

from learning.environments.types import EnvironmentParams
from learning.environments.create_env import create_env
from learning.algorithms.ppo.types import Experiment, Params
from learning.algorithms.ppo.ppo_old import PPO

import pickle as pkl
from pathlib import Path

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

        # Set seeds
        np.random.seed(params.random_seed)
        torch.manual_seed(params.random_seed)
        torch.cuda.manual_seed(params.random_seed)

        env = create_env(
            self.batch_dir,
            env_config.n_envs,
            device=self.device,
            env_name=env_config.environment,
            seed=params.random_seed,
        )

        params.device = self.device
        params.log_filename = self.logs_dir
        params.n_agents = env_config.n_agents
        params.action_dim = env.action_space.spaces[0].shape[0]
        params.state_dim = env.observation_space.spaces[0].shape[0] * params.n_agents
        params.batch_size = env_config.n_envs * params.n_steps
        params.minibatch_size = 64

        learner = PPO(params=params)

        step = 0
        total_episodes = 0
        max_episodes_per_epoch = 10
        max_steps_per_episode = 512
        rmax = -1e10
        running_avg_reward = 0
        data = []
        iterations = 0

        while step < params.n_total_steps:

            rollout_episodes, cum_rewards, state = 0, 0, env.reset()
            episode_len = 0

            for _ in range(0, params.n_steps):

                action = torch.clamp(
                    learner.select_action(
                        self.process_state(
                            env_config.n_envs,
                            state,
                            env_config.state_representation,
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

                learner.add_reward_terminal(reward[0], done)

                cum_rewards += reward[0].item()

                step += 1
                episode_len += 1
                timeout = episode_len == max_steps_per_episode

                if done or timeout:
                    # Log data
                    data.append(cum_rewards)

                    print(f"Step {step}, Reward: {cum_rewards}")

                    running_avg_reward = (
                        0.99 * running_avg_reward + 0.01 * cum_rewards
                        if total_episodes > 0
                        else cum_rewards
                    )

                    # Reset vars, and increase counters
                    state, cum_rewards, episode_len = env.reset(), 0, 0

                    total_episodes += 1
                    rollout_episodes += 1

                    if rollout_episodes == max_episodes_per_epoch:
                        break

            if running_avg_reward > rmax:
                print(f"New best reward: {running_avg_reward} at step {step}")
                rmax = running_avg_reward
                learner.save(self.models_dir / "best_model")

            # if step % 10000 == 0:
            #     learner.save(self.models_dir / f"checkpoint_{step}")

            if iterations % 2 == 0:
                with open(self.models_dir / "data.dat", "wb") as f:
                    pkl.dump(data, f)

            learner.update()

            iterations += 1

    def view(self, exp_config: Experiment, env_config: EnvironmentParams):

        params = Params(**exp_config.params)

        env = create_env(
            self.batch_dir,
            env_config.n_envs,
            device=self.device,
            env_name=env_config.environment,
            seed=params.random_seed,
        )

        params.device = self.device
        params.n_agents = env_config.n_agents
        params.action_dim = env.action_space.spaces[0].shape[0]
        params.state_dim = env.observation_space.spaces[0].shape[0] * params.n_agents

        learner = PPO(params=params)
        learner.load(self.models_dir / "best_model")

        frame_list = []

        n_rollouts = 3

        for i in range(n_rollouts):
            done = False
            state = env.reset()
            R = torch.zeros(env.n_agents)
            r = []
            for t in range(0, params.n_steps):

                action = torch.clamp(
                    learner.deterministic_action(
                        torch.stack(state).permute(1, 0, 2).reshape(1, -1),
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

                r.append(reward)
                R += reward[0]

                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=True,
                )

                frame_list.append(frame)

                if done:
                    print("DONE")
                    break

            print(f"TOTAL RETURN: {R}")
            print(f"MAX {max(r)}")
            print(f"MIN {min(r)}")

        save_video(self.video_name, frame_list, fps=1 / env.scenario.world.dt)
