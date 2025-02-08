import os
import torch


from learning.environments.types import EnvironmentParams
from learning.environments.create_env import create_env
from learning.algorithms.ippo.ppo import Params
from learning.algorithms.ippo.ippo import IPPO
import numpy as np
import pickle as pkl
from pathlib import Path

from vmas.simulator.utils import save_video


class IPPO_Trainer:
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

    def train(self, exp_config, env_config: EnvironmentParams):

        env = create_env(
            self.batch_dir,
            env_config.n_envs,
            device=self.device,
            env_name=env_config.environment,
        )

        params = Params(fname=self.logs_dir, n_agents=env.n_agents)
        params.device = self.device
        params.action_dim = env_config.action_size
        params.state_dim = env_config.observation_size
        params.N_batch = 10
        params.N_steps = 3e6
        params.beta_ent = 0.0
        params.lr_actor = 3e-4
        params.lr_critic = 1e-3
        params.grad_clip = 0.3  # clip_grad_val
        params.write()

        learner = IPPO(params, shared_params=True)
        step = 0
        rmax = -1e10
        running_avg_reward = 0
        data = []
        idx = 0

        while step < params.N_steps:

            for j in range(params.N_batch):
                idx += 1
                done = False
                state = env.reset()
                R = torch.zeros(env_config.n_envs)

                episode_data = []

                while not done:
                    step += 1

                    # Process action and step in environment
                    action = torch.clamp(
                        torch.stack(learner.act(state)), min=-1.0, max=1.0
                    )

                    action = action.reshape(
                        env.n_agents, env_config.n_envs, env_config.action_size
                    )

                    # Uncomment for single agent PPO
                    # action = torch.clamp(
                    #     learner.act(state[0]), min=-1.0, max=1.0
                    # )
                    # action = action.reshape(
                    #     env.n_agents,
                    #     env_config.n_envs,
                    #     env_config.action_size // env.n_agents,
                    # )

                    action_tensor_list = [agent for agent in action]
                    state, reward, done, _ = env.step(action_tensor_list)

                    # Store transition
                    # episode_data.append((state, action, reward, done))

                    learner.add_reward_terminal(reward, done)

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

            learner.train()

    def view(self, exp_config, env_config: EnvironmentParams):

        env = create_env(
            self.batch_dir, 1, device=self.device, env_name=env_config.environment
        )

        params = Params(n_agents=env.n_agents)
        params.device = self.device
        params.action_dim = env_config.action_size
        params.state_dim = env_config.observation_size

        learner = IPPO(params)
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
                    torch.stack(learner.act_deterministic(state)), min=-1.0, max=1.0
                )

                action = action.reshape(env.n_agents, 1, env_config.action_size)

                # Uncomment for single agent PPO
                # action = torch.clamp(
                #     torch.from_numpy(learner.act_deterministic(state[0])),
                #     min=-1.0,
                #     max=1.0,
                # )
                # action = action.reshape(
                #     (env.n_agents, env_config.action_size // env.n_agents)
                # )

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
