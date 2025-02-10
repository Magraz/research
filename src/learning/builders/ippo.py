from learning.algorithms.ippo.types import Experiment, Params
from learning.environments.types import EnvironmentEnum

from dataclasses import asdict

# EXPERIMENT SETTINGS
ENVIRONMENT = EnvironmentEnum.VMAS_SALP
BATCH = f"{ENVIRONMENT}"


# EXPERIMENTS
SINGLE_POLICY = Experiment(params=Params(shared_params=False, single_policy=True))
SHARED_PARAMS = Experiment(params=Params(shared_params=True, single_policy=False))

EXP_DICTS = [
    {
        "batch": f"{ENVIRONMENT}_single_policy",
        "name": f"{ENVIRONMENT}_single_policy",
        "config": asdict(SINGLE_POLICY),
    },
    {
        "batch": f"{ENVIRONMENT}_shared_params",
        "name": f"{ENVIRONMENT}_shared_params",
        "config": asdict(SHARED_PARAMS),
    },
]
