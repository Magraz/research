from learning.algorithms.ippo.types import Experiment, Params
from learning.environments.types import EnvironmentEnum

from dataclasses import asdict

# EXPERIMENT SETTINGS
ENVIRONMENT = EnvironmentEnum.BOX2D_SALP
BATCH_NAME = f"{ENVIRONMENT}_test"
EXPERIMENTS_LIST = ["default"]
DEVICE = "cpu"

# EXPERIMENTS
experiments = []
for experiment_name in EXPERIMENTS_LIST:
    experiment = Experiment(
        device=DEVICE,
        model=experiment_name,
        params=Params(
            n_epochs=10,
            n_total_steps=1e7,
            n_total_episodes=6e4,
            n_max_steps_per_episode=512,
            n_minibatches=4,
            batch_size=5120,
            eps_clip=0.2,
            grad_clip=0.5,
            gamma=0.99,
            lmbda=0.95,
            ent_coef=1e-3,
            std_coef=0.0,
            lr=1e-4,
            random_seeds=[118, 1234, 8764, 3486, 2487, 5439, 6584, 7894, 523, 69],
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
