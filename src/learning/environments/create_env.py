import os
from pathlib import Path
import yaml

from vmas import make_env
from vmas.simulator.environment import Environment

from learning.environments.rover.rover_domain import RoverDomain
from learning.environments.salp.salp_domain import SalpDomain

from learning.environments.types import EnvironmentEnum


def create_env(
    batch_dir, n_envs: int, device: str, env_name: str, **kwargs
) -> Environment:

    env_file = os.path.join(batch_dir, "_env.yaml")

    with open(str(env_file), "r") as file:
        env_config = yaml.safe_load(file)

    match (env_name):
        case EnvironmentEnum.VMAS_ROVER:

            # Environment arguments
            env_args = {
                # Environment data
                "scenario": RoverDomain(),
                "x_semidim": env_config["map_size"][0],
                "y_semidim": env_config["map_size"][1],
                # Agent data
                "n_agents": len(env_config["agents"]),
                "agents_colors": [
                    agent["color"] if agent.get("color") else "BLUE"
                    for agent in env_config["agents"]
                ],
                "agents_positions": [
                    poi["position"]["coordinates"] for poi in env_config["agents"]
                ],
                "lidar_range": [
                    rover["observation_radius"] for rover in env_config["agents"]
                ][0],
                # POIs data
                "n_targets": len(env_config["targets"]),
                "targets_positions": [
                    poi["position"]["coordinates"] for poi in env_config["targets"]
                ],
                "targets_values": [poi["value"] for poi in env_config["targets"]],
                "targets_types": [poi["type"] for poi in env_config["targets"]],
                "targets_orders": [poi["order"] for poi in env_config["targets"]],
                "targets_colors": [
                    poi["color"] if poi.get("color") else "GREEN"
                    for poi in env_config["targets"]
                ],
                "agents_per_target": [poi["coupling"] for poi in env_config["targets"]][
                    0
                ],
                "covering_range": [
                    poi["observation_radius"] for poi in env_config["targets"]
                ][0],
                "use_order": env_config["use_order"],
                "viewer_zoom": kwargs.pop("viewer_zoom", 1),
            }

            # Set up the environment
            env = make_env(
                num_envs=n_envs,
                device=device,
                seed=None,
                # Environment specific variables
                **env_args,
            )

        case EnvironmentEnum.VMAS_SALP:
            env_args = {
                # Environment data
                "scenario": SalpDomain(),
                "x_semidim": env_config["map_size"][0],
                "y_semidim": env_config["map_size"][1],
                # Agent data
                "n_agents": len(env_config["agents"]),
                "agents_colors": [
                    agent["color"] if agent.get("color") else "BLUE"
                    for agent in env_config["agents"]
                ],
                "agents_positions": [
                    poi["position"]["coordinates"] for poi in env_config["agents"]
                ],
                "lidar_range": [
                    rover["observation_radius"] for rover in env_config["agents"]
                ],
                # POIs data
                "n_targets": len(env_config["targets"]),
                "targets_positions": [
                    poi["position"]["coordinates"] for poi in env_config["targets"]
                ],
                "targets_values": [poi["value"] for poi in env_config["targets"]],
                "targets_colors": [
                    poi["color"] if poi.get("color") else "GREEN"
                    for poi in env_config["targets"]
                ],
                "agents_per_target": [poi["coupling"] for poi in env_config["targets"]][
                    0
                ],
                "covering_range": [
                    poi["observation_radius"] for poi in env_config["targets"]
                ][0],
                "viewer_zoom": kwargs.pop("viewer_zoom", 1),
            }

            # Set up the environment
            env = make_env(
                num_envs=n_envs,
                device=device,
                seed=None,
                # Environment specific variables
                **env_args,
            )

    return env
