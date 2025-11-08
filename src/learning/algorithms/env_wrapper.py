from learning.environments.types import EnvironmentEnum
import numpy as np


class EnvWrapper:
    def __init__(self, env, env_name, n_agents):
        self.env = env
        self.env_name = env_name
        self.n_agents = n_agents

    def step(self, actions):
        global_reward = 0

        if (
            self.env_name == EnvironmentEnum.MPE_SPREAD
            or self.env_name == EnvironmentEnum.MPE_SIMPLE
        ):
            next_obs = []
            local_rewards = []
            action_dict = {}

            for i, agent_id in enumerate(self.env.agents):
                action_dict[agent_id] = actions[i][0]

            next_obs, local_rewards, terminated, truncated, info = self.env.step(
                action_dict
            )

            next_obs = np.stack(list(next_obs.values()))
            local_rewards = np.stack(list(local_rewards.values()))
            terminated = np.stack(list(terminated.values()))
            truncated = np.stack(list(truncated.values()))

        else:
            next_obs, global_reward, terminated, truncated, info = self.env.step(
                actions
            )
            local_rewards = np.array(info["local_rewards"])
            terminated = np.array([terminated for _ in range(self.n_agents)])
            truncated = np.array([truncated for _ in range(self.n_agents)])

        return next_obs, global_reward, local_rewards, terminated, truncated, info

    def reset(self):
        if (
            self.env_name == EnvironmentEnum.MPE_SPREAD
            or self.env_name == EnvironmentEnum.MPE_SIMPLE
        ):
            obs, _ = self.env.reset()
            obs = np.stack(list(obs.values()))

        else:
            obs, _ = self.env.reset()

        return obs
