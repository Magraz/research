from learning.algorithms.ppo.types import Experiment, Params
from learning.environments.types import EnvironmentEnum

from dataclasses import asdict

# EXPERIMENT SETTINGS
ENVIRONMENT = EnvironmentEnum.VMAS_SALP
BATCH_NAME = f"{ENVIRONMENT}_global_8a"
EXPERIMENT_NAME = f"mlp"


# EXPERIMENTS
experiment = Experiment(
    device="cpu",
    model="mlp",
    params=Params(
        n_epochs=20,
        n_total_steps=3e6,
        n_steps=5120,
        minibatch_size=128,
        eps_clip=0.2,
        grad_clip=0.5,
        gamma=0.99,
        lmbda=0.95,
        ent_coef=0.01,
        std_coef=0.0,
        lr_actor=3e-4,
        lr_critic=1e-3,
        random_seed=118,
        log_data=True,
    ),
)

EXP_DICTS = [
    {
        "batch": BATCH_NAME,
        "name": EXPERIMENT_NAME,
        "config": asdict(experiment),
    },
]
