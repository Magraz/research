from dataclasses import dataclass


@dataclass
class Params:
    n_steps: int


@dataclass
class Experiment:
    ippo_config: Params = None
