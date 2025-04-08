from learning.algorithms.ppo.types import Experiment, Params
from learning.environments.types import EnvironmentEnum

from dataclasses import asdict

# EXPERIMENT SETTINGS
ENVIRONMENT = EnvironmentEnum.VMAS_SALP
BATCH_NAME = f"{ENVIRONMENT}_single_policy"
EXPERIMENT_NAME = f"transformer"


# EXPERIMENTS
experiment = Experiment(
    params=Params(
        K_epochs=20,
        N_batch=10,
        N_steps=3e6,
        eps_clip=0.2,
        gamma=0.99,
        grad_clip=0.3,
        lmbda=0.95,
        lr_actor=3e-4,
        lr_critic=1e-3,
        random_seed=118,
        beta_ent=0.0,
        log_data=True,
        shared_params=False,
        single_policy=True,
    )
)

EXP_DICTS = [
    {
        "batch": BATCH_NAME,
        "name": EXPERIMENT_NAME,
        "config": asdict(experiment),
    },
]
