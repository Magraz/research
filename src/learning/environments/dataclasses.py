from dataclasses import dataclass


@dataclass(frozen=True)
class PositionConfig:
    spawn_rule: str
    coordinates: tuple[int]


@dataclass(frozen=True)
class AgentsConfig:
    observation_radius: int
    type: int
    color: str
    position: PositionConfig


@dataclass(frozen=True)
class POIConfig:
    value: float
    coupling: int
    observation_radius: float
    type: int
    order: int
    position: PositionConfig


@dataclass(frozen=True)
class EnvironmentConfig:
    map_size: tuple[int]
    agents: list[AgentsConfig]
    targets: list[POIConfig]
    obs_space_dim: int
    action_space_dim: int
