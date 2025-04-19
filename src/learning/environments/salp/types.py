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


class Chain:
    def __init__(self, path: list):
        self.path = path
        self.centroid = self.calculate_centroid()
        # self.orientation = self.calculate_orientation()

    def calculate_centroid(self):

        centroid = self.path.mean(dim=0)

        return centroid

    def calculate_orientation(self):
        """
        Applies the atan2 function to each point's (y, x) coordinates in a tensor of shape (1, n_points, 2).

        Parameters:
            path (torch.Tensor): Tensor of shape (1, n_points, 2), where each entry is (x, y).

        Returns:
            torch.Tensor: Tensor of shape (1, n_points) containing the angles in radians.
        """

        # Extract x and y coordinates
        x = self.path[..., 0] - self.path[:, 0, 0]  # Shape: (1, n_points)
        y = self.path[..., 1] - self.path[:, 0, 1]  # Shape: (1, n_points)

        # Compute the angle using atan2(y, x)
        angles = torch.atan2(y, x)  # Shape: (1, n_points)

        # return angles.mean(dim=1) % (2 * torch.pi)

        orientation = (angles[:, 1:].mean(dim=1) + torch.pi) % (2 * torch.pi) - torch.pi

        if orientation < 0:
            orientation += torch.pi

        return orientation

    # def update(self):
    #     # Update path
    #     for idx, entity in enumerate(self.entities):
    #         self.path[:, idx, :] = entity.state.pos

    #     self.centroid = self.calculate_centroid()
    # self.orientation = self.calculate_orientation()
