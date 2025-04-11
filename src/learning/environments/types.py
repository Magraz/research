from enum import StrEnum
from dataclasses import dataclass


@dataclass
class EnvironmentParams:
    environment: str = None
    agents: list = None
    map_size: tuple[int] = None
    observation_size: int = 0
    action_size: int = 0
    n_envs: int = 1
    n_agents: int = 1
    state_representation: str = None


class EnvironmentEnum(StrEnum):
    VMAS_ROVER = "rover"
    VMAS_SALP = "salp"
    VMAS_BALANCE = "balance"
    VMAS_BUZZ_WIRE = "buzz_wire"
