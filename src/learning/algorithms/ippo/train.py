import os
import torch
from learning.algorithms.types import ExperimentConfig

from learning.environments.types import EnvironmentConfig
from learning.environments.create_env import create_env
from dataclasses import asdict
from pathlib import Path
from learning.algorithms.ippo.PPO import Params
from learning.algorithms.ippo.IPPO import IPPO
import numpy as np
import pickle as pkl


class IPPO_Trainer:
    def __init__(
        self,
        device: str,
        batch_dir: str,
        trials_dir: str,
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
        self.trial_dir = os.path.join(self.trials_dir, self.trial_folder_name)

    def train(self, exp_config: ExperimentConfig, env_config: EnvironmentConfig):
        env = create_env(
            self.batch_dir, 1, device=self.device, env_name=exp_config.environment
        )

        params = Params(fname="save", n_agents=env.n_agents)  # env.n_agents)
        params.device = self.device
        params.action_dim = env_config.action_size
        params.state_dim = env_config.observation_size
        params.action_std = -1.0
        params.beta_ent = 0.0
        params.N_batch = 2
        params.K_epochs = 10
        params.N_steps = 3e6
        params.write()
        learner = IPPO(params)
        step = 0
        rmax = -1e10
        data = []
        idx = 0
        while step < params.N_steps:

            for j in range(params.N_batch):
                idx += 1
                done = False
                state = env.reset()
                R = np.zeros(env.n_agents)
                while not done:
                    step += 1
                    action = learner.act(state)
                    action_tensor_list = [
                        torch.tensor(row).unsqueeze(0) for row in action
                    ]
                    state, reward, done, _ = env.step(action_tensor_list)
                    data.append([state, reward[0]])
                    learner.add_reward_terminal(reward, done)
                    R += np.array(reward[0].cpu())
                print(step, R)

                if rmax < R[0]:
                    # print("Best: "+str(rmax)+"  step: "+str(rmax))
                    learner.save("logs/a0")
                    rmax = R[0]
                learner.save("logs/a1")
            learner.train(step)
            if idx % 10 == 0:
                with open("logs/data.dat", "wb") as f:
                    pkl.dump(data, f)

    def view(self, exp_config: ExperimentConfig, env_config: EnvironmentConfig):
        env = create_env(
            self.batch_dir, 1, device=self.device, env_name=exp_config.environment
        )
        params = Params(n_agents=3)  # env.n_agents)
        params.device = self.device
        params.action_dim = env_config.action_size
        params.state_dim = env_config.observation_size
        learner = IPPO(params)
        learner.load("logs/a1")
        while True:
            done = False
            state = env.reset()
            R = np.zeros(env.n_agents)
            r = []
            while not done:
                action = learner.act_deterministic(state)
                action_tensor_list = [torch.tensor(row).unsqueeze(0) for row in action]
                state, reward, done, _ = env.step(action_tensor_list)
                r.append(reward)
                R += np.array(reward[0].cpu())

                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=True,
                )

            print(R, max(r), min(r))
