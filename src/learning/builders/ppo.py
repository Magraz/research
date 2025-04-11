from learning.algorithms.ppo.types import Experiment, Params
from learning.environments.types import EnvironmentEnum

from dataclasses import asdict

# EXPERIMENT SETTINGS
ENVIRONMENT = EnvironmentEnum.VMAS_SALP
BATCH_NAME = f"{ENVIRONMENT}_single_policy"
EXPERIMENT_NAME = f"mlp"


# EXPERIMENTS
experiment = Experiment(
    params=Params(
        n_epochs=20,
        n_total_steps=3e6,
        n_steps=512,
        n_minibatches=32,
        eps_clip=0.2,
        gamma=0.99,
        grad_clip=0.3,
        lmbda=0.95,
        lr_actor=3e-4,
        lr_critic=1e-3,
        random_seed=118,
        beta_ent=0.001,
        log_data=True,
    )
)

EXP_DICTS = [
    {
        "batch": BATCH_NAME,
        "name": EXPERIMENT_NAME,
        "config": asdict(experiment),
    },
]
