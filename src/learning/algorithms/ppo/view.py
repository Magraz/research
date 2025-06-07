import os
import torch

from learning.environments.types import EnvironmentParams
from learning.environments.create_env import create_env
from learning.algorithms.ppo.types import Experiment, Params
from learning.algorithms.ppo.ppo import PPO
from learning.algorithms.ppo.utils import get_state_dim, process_state

import dill
from pathlib import Path
from statistics import mean
from vmas.simulator.utils import save_video


def view(
    exp_config: Experiment,
    env_config: EnvironmentParams,
    device: str,
    dirs: dict,
    # View parameters
    n_agents_eval=24,
    n_rollouts=250,
    rollout_length=1,
    seed=500,
    render=False,
):

    params = Params(**exp_config.params)

    n_agents_train = env_config.n_agents

    env = create_env(
        dirs["batch"],
        1,
        device,
        env_config.environment,
        seed,  # 10265
        n_agents=n_agents_eval,
        training=False,
    )

    d_action = env.action_space.spaces[0].shape[0]
    d_state = get_state_dim(
        env.observation_space.spaces[0].shape[0],
        env_config.state_representation,
        exp_config.model,
        n_agents_train,
    )

    learner = PPO(
        device,
        exp_config.model,
        params,
        n_agents_train,
        n_agents_eval,
        1,
        d_state,
        d_action,
    )
    learner.load(dirs["models"] / "best_model")
    learner.policy.eval()

    frame_list = []
    info_list = []
    rollout_r_average = []

    for i in range(n_rollouts):

        done = False
        state = env.reset()
        R = 0
        r = []

        for t in range(0, rollout_length):

            action = torch.clamp(
                learner.deterministic_action(
                    process_state(
                        state,
                        env_config.state_representation,
                        exp_config.model,
                    )
                ),
                min=-1.0,
                max=1.0,
            )

            action_tensor = action.reshape(
                1,
                n_agents_eval,
                d_action,
            ).transpose(1, 0)

            # Turn action tensor into list of tensors with shape (n_env, action_dim)
            action_tensor_list = torch.unbind(action_tensor)

            state, reward, done, info = env.step(action_tensor_list)

            info_list.append(info[0])

            r.append(reward[0].item())
            R = reward[0].item()

            if render:
                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=True,
                )

                frame_list.append(frame)

            if torch.any(done):
                print("DONE")
                break

        rollout_r_average.append(R)

        print(f"TOTAL RETURN: {R}")

    print(f"MEAN RETURN OVER {n_rollouts}: {mean(rollout_r_average)}")

    with open(dirs["logs"] / f"test_rollouts_info_{n_agents_eval}.dat", "wb") as f:
        dill.dump(info_list, f)

    if render:
        save_video(
            str(dirs["videos"] / f"view_{n_agents_eval}"),
            frame_list,
            fps=1 / env.scenario.world.dt,
        )
