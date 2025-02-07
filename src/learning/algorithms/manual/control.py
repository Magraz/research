import os
import torch


from learning.environments.types import EnvironmentParams
from learning.environments.create_env import create_env
from learning.algorithms.ippo.ppo import Params
from learning.algorithms.ippo.ippo import IPPO
import numpy as np
import pickle as pkl
from pathlib import Path

from pynput.keyboard import Listener
from learning.testing.manual_control import manual_control


class ManualControl:
    def __init__(
        self,
        device: str,
        batch_dir: Path,
        trials_dir: Path,
        trial_id: int,
        trial_name: str,
        video_name: str,
    ):
        self.n_envs = 1
        # Directories
        self.device = device
        self.batch_dir = batch_dir
        self.trials_dir = trials_dir
        self.trial_name = trial_name
        self.trial_id = trial_id
        self.video_name = video_name

    def view(self, exp_config, env_config: EnvironmentParams):

        env = create_env(
            self.batch_dir,
            self.n_envs,
            device=self.device,
            env_name=env_config.environment,
        )

        mc = manual_control(env.n_agents)

        G_total = torch.zeros((env.n_agents, self.n_envs)).to(self.device)
        G_list = []
        obs_list = []

        with Listener(on_press=mc.on_press, on_release=mc.on_release) as listener:

            listener.join(timeout=1)

            for step in range(env_config.max_steps):

                step += 1

                actions = []

                for i, agent in enumerate(env.agents):

                    # Move one agent at a time
                    if i == mc.controlled_agent:
                        cmd_action = mc.cmd_vel  # + mc.join[:]
                        action = torch.tensor(cmd_action).repeat(self.n_envs, 1)
                    else:
                        action = torch.tensor([0.0, 0.0]).repeat(self.n_envs, 1)

                    # Move all agents at the same time
                    # action = torch.tensor(mc.cmd_vel).repeat(self.n_envs, 1)

                    actions.append(action)

                obs, rews, dones, info = env.step(actions)

                obs_list.append(obs[0][0])

                G_list.append(torch.stack([g[: self.n_envs] for g in rews], dim=0)[0])

                G_total += torch.stack([g[: self.n_envs] for g in rews], dim=0)

                print(G_total)

                G = torch.stack([g[: self.n_envs] for g in rews], dim=0)

                if dones.any():
                    return

                # if any(tensor.any() for tensor in rews):
                #     print("G")
                #     print(G)

                #     # print("Total G")
                #     # print(G_total)

                #     pass

                _ = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=True,
                )
