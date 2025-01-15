import os
import yaml
import torch
from pathlib import Path
from learning.algorithms.ccea.ccea import CooperativeCoevolutionaryAlgorithm
from learning.algorithms.ppo.ppo import PPO
from learning.algorithms.dataclasses import ExperimentConfig
from learning.environments.types import EnvironmentEnum
from learning.environments.rover.dataclasses import RoverEnvironmentConfig

from dataclasses import asdict


def run_algorithm(
    batch_dir: str,
    batch_name: str,
    experiment_name: str,
    algorithm: str,
    environment: str,
    trial_id: int,
):

    exp_file = os.path.join(batch_dir, f"{experiment_name}.yaml")

    with open(str(exp_file), "r") as file:
        exp_dict = yaml.unsafe_load(file)

    env_file = os.path.join(batch_dir, "_env.yaml")

    with open(str(env_file), "r") as file:
        env_dict = yaml.safe_load(file)

    exp_config = ExperimentConfig(**exp_dict)

    match (environment):
        case EnvironmentEnum.VMAS_ROVER:
            env_config = RoverEnvironmentConfig(**env_dict)

        case EnvironmentEnum.VMAS_SALP:
            env_config = RoverEnvironmentConfig(**env_dict)

    match (algorithm):
        case "CCEA":
            algorithm = CooperativeCoevolutionaryAlgorithm(
                batch_dir=batch_dir,
                trials_dir=Path(batch_dir).parents[1]
                / "results"
                / batch_name
                / experiment_name,
                trial_id=trial_id,
                trial_name=Path(exp_file).stem,
                video_name=f"{experiment_name}_{trial_id}",
                device="cuda" if torch.cuda.is_available() else "cpu",
                # Environment Data
                map_size=env_config.map_size,
                observation_size=env_config.obs_space_dim,
                action_size=env_config.action_space_dim,
                n_agents=len(env_config.agents),
                n_pois=len(env_config.targets),
                # Experiment Data
                **asdict(exp_config),
            )
        case "PPO":
            algorithm = PPO(
                batch_dir=batch_dir,
                trials_dir=Path(batch_dir).parents[1]
                / "results"
                / batch_name
                / experiment_name,
                trial_id=trial_id,
                trial_name=Path(exp_file).stem,
                video_name=f"{experiment_name}_{trial_id}",
                device="cuda" if torch.cuda.is_available() else "cpu",
                # Environment Data
                map_size=env_config.map_size,
                observation_size=env_config.obs_space_dim,
                action_size=env_config.action_space_dim,
                n_agents=len(env_config.agents),
                n_pois=len(env_config.targets),
                # Experiment Data
                **asdict(exp_config),
            )

    return algorithm.run()
