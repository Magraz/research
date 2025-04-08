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

    def train(
        self,
        exp_config: Experiment,
        env_config: EnvironmentParams,
    ):
        env = create_env(
            self.batch_dir,
            env_config.n_envs,
            device=self.device,
            env_name=env_config.environment,
        )

        # Set params
        params = Params(**exp_config.params)

        params.device = self.device
        params.log_filename = self.logs_dir
        params.action_dim = env_config.action_size
        params.state_dim = env_config.observation_size

        learner = PPO(params)
        step = 0
        rmax = -1e10
        running_avg_reward = 0
        data = []
        idx = 0

        # Start training loop
        while step < params.N_steps:

            for j in range(params.N_batch):
                idx += 1
                done = False
                state = env.reset()
                R = torch.zeros(env_config.n_envs)

                episode_data = []

                while not done:
                    step += 1

                    batched_state = torch.stack(state)

                    action = torch.clamp(
                        learner.select_action(
                            batched_state.permute(1, 0, 2), n_buffer=0
                        ),
                        min=-1.0,
                        max=1.0,
                    )

                    action_tensor_list = [agent for agent in action.permute(1, 0, 2)]
                    state, reward, done, _ = env.step(action_tensor_list)

                    # Store transition
                    # episode_data.append((state, action, reward, done))

                    learner.add_reward_terminal(reward[0], done, n_buffer=0)

                    R += reward[0]

                # Append episode summary instead of per-step data
                data.append(R.tolist()[0])

                print(step, R)

                running_avg_reward = (
                    0.99 * running_avg_reward + 0.01 * R if step > 0 else R
                )

            if running_avg_reward > rmax:
                print(f"New best reward: {running_avg_reward} at step {step}")
                rmax = running_avg_reward
                learner.save(self.models_dir / "best_model")

            if step % 10000 == 0:
                learner.save(self.models_dir / f"checkpoint_{step}")

            if idx % 2 == 0:
                with open(self.models_dir / "data.dat", "wb") as f:
                    pkl.dump(data, f)

            learner.update()

    def view(self, exp_config: Experiment, env_config: EnvironmentParams):

        env = create_env(
            self.batch_dir, 1, device=self.device, env_name=env_config.environment
        )

        # Set params
        params = Params(**exp_config.params)

        params.device = self.device

        learner = PPO(params)
        learner.load(self.models_dir / "best_model")

        frame_list = []

        n_rollouts = 3

        for i in range(n_rollouts):
            done = False
            state = env.reset()
            R = torch.zeros(env.n_agents)
            r = []
            while not done:

                action = torch.clamp(
                    torch.stack(learner.deterministic_action(state)),
                    min=-1.0,
                    max=1.0,
                )
                action = action.reshape((env.n_agents, 1, env_config.action_size))

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
