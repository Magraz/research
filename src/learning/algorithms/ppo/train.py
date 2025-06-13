import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from learning.environments.types import EnvironmentParams
from learning.environments.create_env import create_env
from learning.algorithms.ppo.types import Experiment, Params
from learning.algorithms.ppo.ppo import PPO
from learning.algorithms.ppo.utils import get_state_dim, process_state

import dill
import random
import time


def train(
    exp_config: Experiment,
    env_config: EnvironmentParams,
    device: str,
    dirs: dict,
    checkpoint: bool = False,
):
    # Set optimal thread settings
    n_threads = 1
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(n_threads)
    print(f"PyTorch using {torch.get_num_threads()} threads")

    # Set logger
    writer = SummaryWriter(dirs["logs"])

    # Set params
    params = Params(**exp_config.params)

    # Set seeds
    np.random.seed(params.random_seed)
    random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)

    # Create environment
    n_envs = env_config.n_envs
    n_agents = env_config.n_agents

    env = create_env(
        dirs["batch"],
        n_envs,
        n_agents=n_agents,
        device=device,
        env_name=env_config.environment,
        seed=params.random_seed,
    )

    # Set state and action dimensions
    d_action = env.action_space.spaces[0].shape[0]
    d_state = get_state_dim(
        env.observation_space.spaces[0].shape[0],
        env_config.state_representation,
        exp_config.model,
        n_agents,
    )

    # Create learner object
    learner = PPO(
        device,
        exp_config.model,
        params,
        n_agents,
        n_agents,
        n_envs,
        d_state,
        d_action,
        writer,
        checkpoint,
    )

    # Checkpoint loading logic
    training_data = {
        "rewards_per_episode": [],
        "steps": [],
        "episodes": [],
        "timestamp": [],
        "dones": [],
    }
    if checkpoint:

        checkpoint_path = dirs["models"] / "checkpoint"

        if checkpoint_path.is_file():
            # Load checkpoint
            learner.load(checkpoint_path)

            # Load data
            with open(dirs["logs"] / "train.dat", "rb") as data_file:
                training_data = dill.load(data_file)

            # Load env up to checkpoint
            with open(dirs["models"] / "env.dat", "rb") as env_file:
                env = dill.load(env_file)

    # Setup loop variables
    global_step = 0
    checkpoint_step = 0
    total_episodes = 0
    running_avg_reward = 0
    rmax = -1e6

    # Log start time
    start_time = time.time()

    while global_step < params.n_total_steps:

        episode_len = torch.zeros(env_config.n_envs, dtype=torch.int32, device=device)
        cum_rewards = torch.zeros(env_config.n_envs, dtype=torch.float32, device=device)

        state = env.reset()

        # Collect batch of data stepping by n_envs
        for _ in range(0, params.batch_size, env_config.n_envs):

            # Clamp because actions are stochastic and can lead to them been out of -1 to 1 range
            b_state, b_action, b_logprob, b_state_val = learner.select_action(
                process_state(
                    state,
                    env_config.state_representation,
                    exp_config.model,
                )
            )
            actions_per_env = torch.clamp(
                b_action,
                min=-1.0,
                max=1.0,
            )

            # Permute action tensor of shape (n_envs, n_agents * action_dim) to (agents, n_env, action_dim)
            action_tensor = actions_per_env.view(n_envs, n_agents, d_action)

            # Turn action tensor into list of tensors with shape (n_env, action_dim)
            action_tensor_list = [action_tensor[:, i] for i in range(n_agents)]

            state, reward, done, _ = env.step(action_tensor_list)

            # Add data to learner buffer
            learner.buffer.add(
                b_state, b_action, b_logprob, b_state_val, reward[0], done
            )

            cum_rewards += reward[0]

            episode_len += torch.ones(
                env_config.n_envs, dtype=torch.int32, device=device
            )

            # Create timeout boolean mask
            timeout = episode_len == params.n_max_steps_per_episode

            # Increase counters
            global_step += env_config.n_envs

            if torch.any(done) or torch.any(timeout):

                # Get done and timeout indices
                done_indices = torch.nonzero(done).flatten().tolist()
                timeout_indices = torch.nonzero(timeout).flatten().tolist()

                # Merge indices and remove duplicates
                indices = list(set(done_indices + timeout_indices))

                for idx in indices:
                    r = cum_rewards[idx].item()

                    # Log data when episode is done
                    training_data["rewards_per_episode"].append(r)

                    running_avg_reward = (
                        0.99 * running_avg_reward + 0.01 * r
                        if total_episodes > 0
                        else r
                    )

                    # Reset vars, and increase counters
                    state = env.reset_at(index=idx)
                    cum_rewards[idx], episode_len[idx] = 0, 0

                    total_episodes += 1

                training_data["dones"].append(len(indices))
                training_data["steps"].append(global_step)
                training_data["episodes"].append(total_episodes)
                training_data["timestamp"].append(time.time() - start_time)

        # Store best model if running average reward is higher than previous bestAdd commentMore actions
        if running_avg_reward > rmax:
            rmax = running_avg_reward
            learner.save(dirs["models"] / "best_model")

        # Store checkpoint
        if global_step - checkpoint_step >= 10000:
            # Save model
            learner.save(dirs["models"] / "checkpoint")

            # Store reward per episode data
            with open(dirs["logs"] / "train.dat", "wb") as f:
                dill.dump(training_data, f)

            # Store environment
            with open(dirs["models"] / "env.dat", "wb") as f:
                dill.dump(env, f)

            checkpoint_step = global_step

        print(
            f"Step: {global_step}, Episodes: {total_episodes}, Running Avg Reward: {running_avg_reward}, Minutes {'{:.2f}'.format((time.time() - start_time) / 60)}"
        )

        # Do training step
        learner.update()

        # Log reward data with tensorboard
        if writer is not None:
            for reward in training_data["rewards_per_episode"]:
                writer.add_scalar("Agent/rewards_per_episode", reward, total_episodes)

    writer.close()
