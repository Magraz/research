from learning.algorithms.dataclasses import (
    ExperimentConfig,
    PolicyConfig,
    CCEAConfig,
)
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

#EXPERIMENT SETTINGS
ENVIRONMENT = EnvironmentEnum.ROVER
BATCH = f"{ENVIRONMENT}_static_spread"

#POLICY SETTINGS
GRU_POLICY_LAYERS = [83]
MLP_POLICY_LAYERS = [64, 64]
OUTPUT_MULTIPLIER = 1.0
WEIGHT_INITIALIZATION = InitializationEnum.KAIMING

#CCEA SETTINGS
N_STEPS = 100
N_GENS = 5000
SUBPOP_SIZE = 100
N_GENS_BETWEEN_SAVE = 10
FITNESS_CALC = FitnessCalculationEnum.LAST
MEAN = 0.0
MIN_STD_DEV = 0.05
MAX_STD_DEV = 0.25

GRU_POLICY_CONFIG = PolicyConfig(
    type=PolicyEnum.GRU,
    weight_initialization=WEIGHT_INITIALIZATION,
    hidden_layers=GRU_POLICY_LAYERS,
    output_multiplier=OUTPUT_MULTIPLIER,
)

MLP_POLICY_CONFIG = PolicyConfig(
    type=PolicyEnum.MLP,
    weight_initialization=WEIGHT_INITIALIZATION,
    hidden_layers=MLP_POLICY_LAYERS,
    output_multiplier=OUTPUT_MULTIPLIER,
)

G_CCEA = CCEAConfig(
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
)

D_CCEA = deepcopy(G_CCEA)
D_CCEA.fitness_shaping = FitnessShapingEnum.D

# EXPERIMENTS
G_MLP = ExperimentConfig(
    environment=ENVIRONMENT,
    n_gens_between_save=N_GENS_BETWEEN_SAVE,
    policy_config=MLP_POLICY_CONFIG,
    ccea_config=G_CCEA,
)

G_GRU = deepcopy(G_MLP)
G_GRU.policy_config = GRU_POLICY_CONFIG

EXP_DICTS = [
    {"name": "g_gru", "config": asdict(G_GRU)},
    {"name": "g_mlp", "config": asdict(G_MLP)},
]
