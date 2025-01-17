from enum import StrEnum
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentConfig:
    agents: list
    map_size: tuple[int]
    observation_size: int
    action_size: int


class EnvironmentEnum(StrEnum):
    VMAS_ROVER = "rover"
    VMAS_SALP = "salp"
