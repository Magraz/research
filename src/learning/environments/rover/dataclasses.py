from dataclasses import dataclass
from learning.environments.dataclasses import EnvironmentConfig


@dataclass(frozen=True)
class RoverEnvironmentConfig(EnvironmentConfig):
    use_order: bool
