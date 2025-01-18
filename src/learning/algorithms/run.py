import os
import yaml
import torch
from pathlib import Path

from learning.algorithms.ccea.train import CCEA_Trainer
from learning.algorithms.ccea.types import CCEA_ExperimentConfig

from learning.algorithms.ippo.train import IPPO_Trainer

from learning.algorithms.types import ExperimentConfig, AlgorithmEnum

from learning.environments.types import EnvironmentEnum, EnvironmentConfig
from learning.environments.rover.types import RoverEnvironmentConfig
from learning.environments.salp.types import SalpEnvironmentConfig

from dataclasses import asdict


def run_algorithm(
    batch_dir: str,
    batch_name: str,
    experiment_name: str,
    algorithm: str,
    environment: str,
    trial_id: int,
):

    env_file = os.path.join(batch_dir, "_env.yaml")

    with open(str(env_file), "r") as file:
        env_dict = yaml.safe_load(file)

    # Load environment config
    match (environment):
        case EnvironmentEnum.VMAS_ROVER:
            env_config = RoverEnvironmentConfig(**env_dict)

        case EnvironmentEnum.VMAS_SALP:
            env_config = SalpEnvironmentConfig(**env_dict)

        case EnvironmentEnum.VMAS_BALANCE:
            env_config = EnvironmentConfig(**env_dict)

    # Load experiment config
    exp_file = os.path.join(batch_dir, f"{experiment_name}.yaml")

    with open(str(exp_file), "r") as file:
        exp_dict = yaml.unsafe_load(file)

    match (algorithm):

        case AlgorithmEnum.CCEA:
            exp_config = CCEA_ExperimentConfig(**exp_dict)
            trainer = CCEA_Trainer(
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_dir=batch_dir,
                trials_dir=Path(batch_dir).parents[1]
                / "results"
                / batch_name
                / experiment_name,
                trial_id=trial_id,
                trial_name=Path(exp_file).stem,
                video_name=f"{experiment_name}_{trial_id}",
            )

        case AlgorithmEnum.IPPO:
            exp_config = ExperimentConfig(**exp_dict)
            trainer = IPPO_Trainer(
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_dir=batch_dir,
                trials_dir=Path(batch_dir).parents[1]
                / "results"
                / batch_name
                / experiment_name,
                trial_id=trial_id,
                trial_name=Path(exp_file).stem,
                video_name=f"{experiment_name}_{trial_id}",
            )

    return trainer.train(  # Environment Data
        env_config=env_config,
        # Experiment Data
        exp_config=exp_config,
    )
