from dataclasses import dataclass
from enum import StrEnum


@dataclass
class ExperimentConfig:
    environment: str = ""


class AlgorithmEnum(StrEnum):
    CCEA = "CCEA"
    IPPO = "ippo"
