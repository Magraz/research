from learning.algorithms.ccea.types import (
    CCEA_ExperimentConfig,
)
from learning.algorithms.ccea.types import CCEA_Config, CCEA_PolicyConfig

from learning.algorithms.ccea.types import (
    FitnessShapingEnum,
    FitnessCalculationEnum,
    SelectionEnum,
    PolicyEnum,
    InitializationEnum,
)

from learning.environments.types import EnvironmentEnum

from copy import deepcopy
from dataclasses import asdict

# EXPERIMENT SETTINGS
ENVIRONMENT = EnvironmentEnum.VMAS_SALP
BATCH = f"{ENVIRONMENT}_static_spread"

# POLICY SETTINGS
GRU_POLICY_LAYERS = [83]
MLP_POLICY_LAYERS = [64, 64]
OUTPUT_MULTIPLIER = 1.0
WEIGHT_INITIALIZATION = InitializationEnum.KAIMING

# CCEA SETTINGS
N_STEPS = 100
N_GENS = 5000
SUBPOP_SIZE = 100
N_GENS_BETWEEN_SAVE = 10
FITNESS_CALC = FitnessCalculationEnum.LAST
MEAN = 0.0
MIN_STD_DEV = 0.05
MAX_STD_DEV = 0.25

GRU_POLICY_CONFIG = CCEA_PolicyConfig(
    type=PolicyEnum.GRU,
    weight_initialization=WEIGHT_INITIALIZATION,
    hidden_layers=GRU_POLICY_LAYERS,
    output_multiplier=OUTPUT_MULTIPLIER,
)

MLP_POLICY_CONFIG = CCEA_PolicyConfig(
    type=PolicyEnum.MLP,
    weight_initialization=WEIGHT_INITIALIZATION,
    hidden_layers=MLP_POLICY_LAYERS,
    output_multiplier=OUTPUT_MULTIPLIER,
)

G_CCEA_MLP = CCEA_Config(
    n_steps=N_STEPS,
    n_gens=N_GENS,
    fitness_shaping=FitnessShapingEnum.G,
    selection=SelectionEnum.SOFTMAX,
    subpopulation_size=SUBPOP_SIZE,
    fitness_calculation=FITNESS_CALC,
    mutation={
        "mean": MEAN,
        "min_std_deviation": MIN_STD_DEV,
        "max_std_deviation": MAX_STD_DEV,
    },
    policy_config=MLP_POLICY_CONFIG,
)

D_CCEA_MLP = deepcopy(G_CCEA_MLP)
D_CCEA_MLP.fitness_shaping = FitnessShapingEnum.D

G_CCEA_GRU = deepcopy(G_CCEA_MLP)
G_CCEA_GRU.policy_config = GRU_POLICY_CONFIG

D_CCEA_GRU = deepcopy(D_CCEA_MLP)
D_CCEA_GRU.policy_config = GRU_POLICY_CONFIG


# EXPERIMENTS
G_MLP = CCEA_ExperimentConfig(
    environment=ENVIRONMENT,
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    ccea_config=G_CCEA_MLP,
)

D_MLP = CCEA_ExperimentConfig(
    environment=ENVIRONMENT,
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    ccea_config=D_CCEA_MLP,
)

G_GRU = deepcopy(G_MLP)
G_GRU.ccea_config = G_CCEA_GRU

D_GRU = deepcopy(D_MLP)
D_GRU.ccea_config = D_CCEA_GRU

EXP_DICTS = [
    {"name": "d_mlp", "config": asdict(D_MLP)},
    {"name": "g_mlp", "config": asdict(G_MLP)},
]
