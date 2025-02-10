from dataclasses import dataclass


@dataclass
class Params:
    shared_params: bool
    single_policy: bool


@dataclass
class Experiment:
    params: Params = None
