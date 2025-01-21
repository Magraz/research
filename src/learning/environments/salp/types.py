from dataclasses import dataclass
from learning.environments.types import EnvironmentParams


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


@dataclass
class SalpEnvironmentConfig(EnvironmentParams):
    agents: list[SalpsConfig] = None
    targets: list[POIConfig] = None
