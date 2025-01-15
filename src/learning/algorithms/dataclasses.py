from learning.algorithms.ccea.dataclasses import CCEAConfig
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyConfig:
    weight_initialization: str
    type: str
    hidden_layers: tuple[int]
    output_multiplier: float


@dataclass
class ExperimentConfig:
    environment: str = ""
    use_teaming: bool = False
    ccea_config: CCEAConfig = None
    policy_config: PolicyConfig = None
    n_gens_between_save: int = 0
    team_size: int = 0
