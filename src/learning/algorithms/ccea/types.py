from enum import StrEnum
from dataclasses import dataclass


@dataclass(frozen=True)
class CCEA_PolicyConfig:
    weight_initialization: str
    type: str
    hidden_layers: tuple[int]
    output_multiplier: float


@dataclass
class CCEA_Config:
    n_gens: int
    n_steps: int
    subpopulation_size: int
    selection: str
    fitness_shaping: str
    fitness_calculation: str
    mutation: dict
    policy_config: CCEA_PolicyConfig


class Team:
    def __init__(
        self,
        individuals: list = None,
        combination: list = None,
    ):
        self.individuals = individuals if individuals is not None else []
        self.combination = combination if combination is not None else []


class EvalInfo:
    def __init__(
        self,
        team: Team,
        team_fitness: float,
        agent_fitnesses: list[float],
    ):
        self.team = team
        self.agent_fitnesses = agent_fitnesses
        self.team_fitness = team_fitness


class InitializationEnum(StrEnum):
    KAIMING = "kaiming"


class CCEA_PolicyEnum(StrEnum):
    GRU = "GRU"
    MLP = "MLP"
    CNN = "CNN"


class FitnessShapingEnum(StrEnum):
    D = "difference"
    G = "global"
    HOF = "hof_difference"
    FC = "fitness_critics"


class FitnessCriticError(StrEnum):
    MSE = "MSE"
    MAE = "MAE"
    MSE_MAE = "MSE+MAE"


class FitnessCriticType(StrEnum):
    MLP = "MLP"
    GRU = "GRU"
    ATT = "ATT"


class SelectionEnum(StrEnum):
    SOFTMAX = "softmax"
    EPSILON = "epsilon"
    BINARY = "binary"
    TOURNAMENT = "tournament"


class FitnessCalculationEnum(StrEnum):
    AGG = "aggregate"
    LAST = "last_step"
