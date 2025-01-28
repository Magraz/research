from learning.algorithms.ippo.types import Experiment
from learning.environments.types import EnvironmentEnum

from dataclasses import asdict

# EXPERIMENT SETTINGS
ENVIRONMENT = EnvironmentEnum.VMAS_SALP
BATCH = f"{ENVIRONMENT}"

# # EXPERIMENTS
IPPO = Experiment()

EXP_DICTS = [
    {"name": "ippo", "config": asdict(IPPO)},
]
