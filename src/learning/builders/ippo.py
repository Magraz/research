from learning.algorithms.ippo.types import Experiment, Params
from learning.environments.types import EnvironmentEnum

from dataclasses import asdict

# EXPERIMENT SETTINGS
ENVIRONMENT = EnvironmentEnum.MPE_SIMPLE
BATCH_NAME = f"{ENVIRONMENT}_test"
EXPERIMENTS_LIST = ["mlp", "gru"]
# EXPERIMENTS_LIST = ["mlp_shared", "gru_shared"]
DEVICE = "cpu"
MODELS = ["mlp", "gru"]

# EXPERIMENTS
experiments = []
for i, experiment_name in enumerate(EXPERIMENTS_LIST):
    experiment = Experiment(
        device=DEVICE,
        model=MODELS[i],
        params=Params(
            n_epochs=4,
            n_total_steps=2e7,
            n_minibatches=4,
            batch_size=250,
            parameter_sharing=True,
            random_seeds=[118, 1234, 8764, 3486, 2487, 5439, 6584, 7894, 523, 69],
            eps_clip=0.2,
            grad_clip=0.5,
            gamma=0.99,
            lmbda=0.95,
            ent_coef=1e-2,
            val_coef=0.5,
            std_coef=0.0,
            lr=3e-4,
        ),
    )
    experiments.append(experiment)

EXP_DICTS = [
    {
        "batch": BATCH_NAME,
        "name": EXPERIMENTS_LIST[i],
        "config": asdict(experiment),
    }
    for i, experiment in enumerate(experiments)
]
