from dataclasses import dataclass
from learning.environments.types import EnvironmentParams
from vmas.simulator.core import Entity
import torch


@dataclass(frozen=True)
class Position:
    spawn_rule: str
    coordinates: tuple[int]


@dataclass(frozen=True)
class Agent:
    observation_radius: int
    starting_position: tuple[int]
    n_agents: int


@dataclass(frozen=True)
class Target:
    value: float
    observation_radius: float
    position: Position


@dataclass
class SalpEnvironmentParams(EnvironmentParams):
    agents: list[Agent] = None
    targets: list[Target] = None
    state_representation: str = None
    shuffle_agents_positions: bool = False


class Chain:
    def __init__(self, idx: int, path: list, entities: list[Entity]):
        self.idx = idx
        self.path = path
        self.entities = entities
        self.centroid = self.calculate_centroid()

    def calculate_centroid(self):
        """
        Calculate the centroid of a list of (x, y) tuples.

        Parameters:
            points (list): List of (x, y) tuples.

        Returns:
            tuple: The centroid as (x_c, y_c).
        """

        # # Calculate the centroid
        # n_points = self.path.shape[1]
        # x_c = torch.sum(self.path[:, :, 0], dim=1) / n_points
        # y_c = torch.sum(self.path[:, :, 1], dim=1) / n_points

        # centroid = torch.stack((x_c, y_c), dim=1)

        temp_centroid = self.path.mean(dim=1)

        return temp_centroid

    def update(self):
        # Update path
        for idx, entity in enumerate(self.entities):
            self.path[:, idx, :] = entity.state.pos

        self.centroid = self.calculate_centroid()
