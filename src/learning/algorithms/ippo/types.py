from dataclasses import dataclass
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Params:
    shared_params: bool = False
    single_policy: bool = False
    K_epochs: int = 0
    N_batch: int = 0
    N_steps: int = 0
    eps_clip: float = 0.0
    gamma: float = 0.0

    lr_actor: float = 0.0
    lr_critic: float = 0.0
    action_std: float = 0.0
    random_seed: int = 0
    grad_clip: float = 0.0

    action_dim: int = 0
    state_dim: int = 0

    actor_hidden: int = 0
    critic_hidden: int = 0
    active_fn: nn = None

    lmbda: float = 0.0

    device: str = ""
    log_indiv: bool = False

    writer: SummaryWriter = None
    log_indiv: bool = False

    beta_ent: float = 0.0
    n_agents: int = 0


@dataclass
class Experiment:
    params: Params = None
