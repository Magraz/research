from dataclasses import dataclass


@dataclass
class Params:

    n_epochs: int
    n_total_steps: int
    n_max_steps_per_episode: int
    batch_size: int
    minibatch_size: int

    eps_clip: float
    gamma: float

    lr_critic: float
    lr_actor: float
    random_seed: int
    grad_clip: float
    ent_coef: float
    std_coef: float
    lmbda: float
    log_data: bool


@dataclass
class Experiment:
    device: str = ""
    model: str = ""
    params: Params = None
