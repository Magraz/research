from dataclasses import dataclass
from learning.environments.types import EnvironmentConfig


@dataclass(frozen=True)
class PositionConfig:
    spawn_rule: str
    coordinates: tuple[int]


@dataclass(frozen=True)
class SalpsConfig:
    observation_radius: int
    type: int
    color: str
    position: PositionConfig


@dataclass(frozen=True)
class POIConfig:
    value: float
    coupling: int
    observation_radius: float
    position: PositionConfig


@dataclass(frozen=True)
class SalpEnvironmentConfig(EnvironmentConfig):
    agents: list[SalpsConfig]
    targets: list[POIConfig]
