import torch
from learning.algorithms.ippo.trainer import IPPOTrainer
from learning.environments.types import EnvironmentParams
from learning.algorithms.ippo.types import Experiment, Params
import random
import numpy as np


def set_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed_all(seed)  # PyTorch CUDA
    torch.backends.cudnn.deterministic = True  # Make CUDA deterministic
    torch.backends.cudnn.benchmark = False  # Disable CUDA benchmarking


def train(
    exp_config: Experiment,
    env_config: EnvironmentParams,
    device: str,
    trial_id: str,
    dirs: dict,
    checkpoint: bool = False,
):

    # Set params
    params = Params(**exp_config.params)

    # Set seeds
    random_seed = params.random_seeds[0]

    if trial_id.isdigit():
        random_seed = params.random_seeds[int(trial_id)]

    # Set all random seeds for reproducibility
    set_seeds(random_seed)

    # Device configuration
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {exp_config.device}")

    # Environment configuration
    env_config = {
        "render_mode": None,  # Set to "human" for visual training
        "n_agents": env_config.n_agents,
    }

    # PPO configuration
    ppo_config = {
        "lr": params.lr,
        "gamma": params.gamma,
        "gae_lambda": params.lmbda,
        "clip_epsilon": params.eps_clip,
        "entropy_coef": params.ent_coef,
    }

    # Create trainer
    trainer = IPPOTrainer(dirs, env_config, ppo_config, device)

    # Train
    trainer.train(total_steps=params.n_total_steps, batch_size=params.batch_size)
    trainer.save_training_stats(dirs["logs"] / "training_stats_finished.pkl")

    # Save trained agents
    trainer.save_agents(dirs["models"] / "models_finished.pth")

    # Test trained agents with rendering
    print("\nTesting trained agents...")
    trainer.env.render_mode = "human"
    trainer.render_episode()
    trainer.env.close()
