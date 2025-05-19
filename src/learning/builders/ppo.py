from learning.algorithms.ppo.types import Experiment, Params
from learning.environments.types import EnvironmentEnum

from dataclasses import asdict

# EXPERIMENT SETTINGS
ENVIRONMENT = EnvironmentEnum.VMAS_SALP
BATCH_NAME = f"{ENVIRONMENT}_local_8a"
EXPERIMENTS_LIST = [
    "transformer",
    "transformer_full",
    "transformer_encoder",
    "transformer_decoder",
    "gat",
    "gcn",
    "mlp",
]
DEVICE = "cpu"

# EXPERIMENTS
experiments = []
for experiment_name in EXPERIMENTS_LIST:
    experiment = Experiment(
        device=DEVICE,
        model=experiment_name,
        params=Params(
            n_epochs=10,
            n_total_steps=1e8,
            n_max_steps_per_episode=512,
            batch_size=5120,
            minibatch_size=256,
            eps_clip=0.2,
            grad_clip=0.5,
            gamma=0.99,
            lmbda=0.95,
            ent_coef=1e-3,
            std_coef=0.0,
            lr_actor=5e-5,
            lr_critic=5e-5,
            random_seed=118,
        ),
    )
    experiments.append(experiment)

EXP_DICTS = [
    {
        "batch": BATCH_NAME,
        "name": experiment.model,
        "config": asdict(experiment),
    }
    for experiment in experiments
]
