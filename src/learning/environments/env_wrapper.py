import numpy as np


class EnvWrapper:
    """Wrapper around the Environment to expose a cleaner interface for RL
    Parameters:
        env_name (str): Env name
    """

    def __init__(self, args, num_envs, env):
        """
        A base template for all environment wrappers.
        """
        # Initialize world with requiste params
        self.args = args

        self.universe = []  # Universe - collection of all envs running in parallel

    def reset(self):
        """Method overloads reset
        Parameters:
            None

        Returns:
            next_obs (list): Next state
        """
        joint_obs = []
        for env_id, env in enumerate(self.universe):
            obs = env.reset()

            joint_obs.append(obs)

        joint_obs = np.stack(joint_obs, axis=1)

        # returns [agent_id, universe_id, obs]

        return joint_obs

    def step(self, action):  # Expects a numpy action
        """Take an action to forward the simulation

        Parameters:
            action (ndarray): action to take in the env

        Returns:
            next_obs (list): Next state
            reward (float): Reward for this step
            done (bool): Simulation done?
            info (None): Template from OpenAi gym (doesnt have anything)
        """

        joint_obs = []
        joint_reward = []
        joint_done = []
        joint_global = []

        for universe_id, env in enumerate(self.universe):
            if not self.args.config.ngu:
                next_state, reward, done, info = env.step(action[:, universe_id, :])
            else:
                next_state, reward, done, info = env.step(action[:, universe_id, :])

            joint_obs.append(next_state)
            joint_reward.append(reward)
            joint_done.append(done)
            joint_global.append(info)

        joint_obs = np.stack(joint_obs, axis=1)
        joint_reward = np.stack(joint_reward, axis=1)

        return joint_obs, joint_reward, joint_done, joint_global
