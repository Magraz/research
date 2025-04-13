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

        n_batches = 10

        params.device = self.device
        params.log_filename = self.logs_dir
        params.n_agents = env_config.n_agents
        params.action_dim = env.action_space.spaces[0].shape[0]
        params.state_dim = env.observation_space.spaces[0].shape[0] * params.n_agents
        params.batch_size = env_config.n_envs * params.n_steps
        params.minibatch_size = params.batch_size // params.n_minibatches

        learner = PPO(params=params)

        step = 0
        episodes = 0
        rmax = -1e10
        running_avg_reward = 0
        data = []
        iterations = 0

        while step < params.n_total_steps:

            for j in range(n_batches):
                done = False
                state = env.reset()
                R = 0.0

                for t in range(0, params.n_steps):

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

                    learner.buffer.rewards.append(reward[0])
                    learner.buffer.is_terminals.append(done)

                    R += reward[0].item()

                    step += 1

                    if done:
                        break

                # Append cumulative reward per episode
                data.append(R)

                print(f"Step {step}, Reward: {R}")

                running_avg_reward = (
                    0.99 * running_avg_reward + 0.01 * R if episodes > 0 else R
                )

                episodes += 1

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

        learner = PPO(
            state_dim=params.state_dim,
            action_dim=params.action_dim * params.n_agents,
            lr_actor=params.lr_actor,
            lr_critic=params.lr_critic,
            gamma=params.gamma,
            K_epochs=params.n_epochs,
            eps_clip=params.eps_clip,
            device=params.device,
        )
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
                    learner.select_action(
                        torch.stack(state).permute(1, 0, 2).reshape(1, -1)
                    ),
                    min=-1.0,
                    max=1.0,
                )

                action = action.reshape(
                    params.n_agents,
                    env_config.n_envs,
                    params.action_dim,
                )

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
