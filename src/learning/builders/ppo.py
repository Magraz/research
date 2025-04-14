from learning.algorithms.ppo.types import Experiment, Params
from learning.environments.types import EnvironmentEnum

from dataclasses import asdict

# EXPERIMENT SETTINGS
ENVIRONMENT = EnvironmentEnum.VMAS_BUZZ_WIRE
BATCH_NAME = f"{ENVIRONMENT}_standard"
EXPERIMENT_NAME = f"mlp"


# EXPERIMENTS
experiment = Experiment(
    params=Params(
        n_epochs=20,
        n_total_steps=1e6,
        n_steps=512,
        n_minibatches=16,
        eps_clip=0.2,
        grad_clip=0.5,
        gamma=0.99,
        lmbda=0.95,
        ent_coef=1e-3,
        std_coef=1e-2,
        lr_actor=3e-4,
        lr_critic=1e-3,
        random_seed=118,
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
