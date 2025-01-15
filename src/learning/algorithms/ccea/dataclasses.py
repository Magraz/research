from dataclasses import dataclass


@dataclass
class CCEAConfig:
    n_gens: int
    n_steps: int
    subpopulation_size: int
    selection: str
    fitness_shaping: str
    fitness_calculation: str
    mutation: dict
