import os
import yaml
import torch
from pathlib import Path

from learning.algorithms.ccea.train import CCEA_Trainer
from learning.algorithms.ccea.types import Experiment as CCEA_Experiment

from learning.algorithms.ppo.train import PPO_Trainer
from learning.algorithms.td3.train import TD3_Trainer

from learning.algorithms.ippo.train import IPPO_Trainer
from learning.algorithms.ppo.types import Experiment as PPO_Experiment

from learning.algorithms.manual.control import ManualControl

from learning.algorithms.types import AlgorithmEnum

from learning.environments.types import EnvironmentEnum, EnvironmentParams
from learning.environments.rover.types import RoverEnvironmentParams
from learning.environments.salp.types import SalpEnvironmentParams


def run_algorithm(
    batch_dir: Path,
    batch_name: str,
    experiment_name: str,
    algorithm: str,
    environment: str,
    trial_id: str,
    train: bool,
    checkpoint: bool,
):

    env_file = batch_dir / "_env.yaml"

    with open(env_file, "r") as file:
        env_dict = yaml.safe_load(file)

    # Load environment config
    match (environment):
        case EnvironmentEnum.VMAS_ROVER:
            env_config = RoverEnvironmentParams(**env_dict)

        case EnvironmentEnum.VMAS_SALP:
            env_config = SalpEnvironmentParams(**env_dict)

        case EnvironmentEnum.VMAS_BALANCE | EnvironmentEnum.VMAS_BUZZ_WIRE:
            env_config = EnvironmentParams(**env_dict)

    env_config.environment = environment

    # Load experiment config
    exp_file = batch_dir / f"{experiment_name}.yaml"

    with open(exp_file, "r") as file:
        exp_dict = yaml.unsafe_load(file)

    match (algorithm):

        case AlgorithmEnum.CCEA:
            exp_config = CCEA_Experiment(**exp_dict)
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
            exp_config = PPO_Experiment(**exp_dict)
            trainer = IPPO_Trainer(
                device="cpu",
                batch_dir=batch_dir,
                trials_dir=Path(batch_dir).parents[1]
                / "results"
                / batch_name
                / experiment_name,
                trial_id=trial_id,
                trial_name=Path(exp_file).stem,
                video_name=f"{experiment_name}_{trial_id}",
            )

        case AlgorithmEnum.PPO:
            exp_config = PPO_Experiment(**exp_dict)
            trainer = PPO_Trainer(
                device=exp_config.device,
                batch_dir=batch_dir,
                trials_dir=Path(batch_dir).parents[1]
                / "results"
                / batch_name
                / experiment_name,
                trial_id=trial_id,
                video_name=f"{experiment_name}_{trial_id}",
                checkpoint=checkpoint,
            )

        case AlgorithmEnum.TD3:
            exp_config = None
            trainer = TD3_Trainer(
                device="cpu",
                batch_dir=batch_dir,
                trials_dir=Path(batch_dir).parents[1]
                / "results"
                / batch_name
                / experiment_name,
                trial_id=trial_id,
                trial_name=Path(exp_file).stem,
                video_name=f"{experiment_name}_{trial_id}",
            )

        case AlgorithmEnum.NONE:
            exp_config = None
            train = False

            trainer = ManualControl(
                device="cpu",
                batch_dir=batch_dir,
                trials_dir=Path(batch_dir).parents[1]
                / "results"
                / batch_name
                / experiment_name,
                trial_id=trial_id,
                trial_name=Path(exp_file).stem,
                video_name=f"{experiment_name}_{trial_id}",
            )

    if train:
        trainer.train(  # Environment Data
            env_config=env_config,
            # Experiment Data
            exp_config=exp_config,
        )
    else:
        trainer.view(  # Environment Data
            env_config=env_config,
            # Experiment Data
            exp_config=exp_config,
        )
