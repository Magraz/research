from dataclasses import dataclass
import torch.nn as nn


@dataclass
class Params:

    shared_params: bool
    single_policy: bool
    K_epochs: int
    N_batch: int
    N_steps: int
    eps_clip: float
    gamma: float

    lr_critic: float
    lr_actor: float
    random_seed: int
    grad_clip: float
    beta_ent: float
    lmbda: float
    log_data: bool

    # Default params
    device: str = ""
    n_agents: int = 0
    log_filename: str = ""
    action_dim: int = 0
    state_dim: int = 0


@dataclass
class Experiment:
    params: Params = None
