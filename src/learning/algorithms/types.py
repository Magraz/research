from dataclasses import dataclass
from learning.algorithms.ccea.types import CCEA_Config


@dataclass
class ExperimentConfig:
    environment: str = ""
    n_gens_between_save: int = 0
    ccea_config: CCEA_Config = None
