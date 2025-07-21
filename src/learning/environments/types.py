from enum import StrEnum
from dataclasses import dataclass


@dataclass
class EnvironmentParams:
    environment: str = None
    n_envs: int = 1
    n_agents: int = 1
    state_representation: str = None


class EnvironmentEnum(StrEnum):
    VMAS_ROVER = "rover"
    VMAS_SALP_NAVIGATE = "salp_navigate"
    VMAS_SALP_PASSAGE = "salp_passage"
    VMAS_BALANCE = "balance"
    VMAS_BUZZ_WIRE = "buzz_wire"
    MAMUJOCO_SWIMMER = "many_segment_swimmer"
