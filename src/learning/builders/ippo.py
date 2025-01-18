from learning.algorithms.types import (
    ExperimentConfig,
)
from learning.environments.types import EnvironmentEnum

from dataclasses import asdict

# EXPERIMENT SETTINGS
ENVIRONMENT = EnvironmentEnum.VMAS_BALANCE
BATCH = f"{ENVIRONMENT}_standard"

# EXPERIMENTS
IPPO = ExperimentConfig(environment=ENVIRONMENT)

EXP_DICTS = [
    {"name": "ippo", "config": asdict(IPPO)},
]
