import torch
from learning.algorithms.ippo.trainer import IPPOTrainer
from learning.environments.types import EnvironmentParams, EnvironmentEnum
from learning.algorithms.ippo.types import Experiment, Params
from learning.environments.box2d_salp.domain import SalpChainEnv

from mpe2 import simple_spread_v3

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

    # PPO configuration
    ppo_config = {
        "lr": params.lr,
        "gamma": params.gamma,
        "gae_lambda": params.lmbda,
        "clip_epsilon": params.eps_clip,
        "entropy_coef": params.ent_coef,
    }

    # Create environment
    using_hybrid_actions = False
    match (env_config.environment):
        case EnvironmentEnum.BOX2D_SALP:
            # Environment configuration
            env_config = {
                "render_mode": None,  # Set to "human" for visual training
                "n_agents": env_config.n_agents,
            }
            env = SalpChainEnv(**env_config)
            state_dim = env.observation_space.shape[1]
            action_dim = env.action_space.shape[1]
            # Check if we're dealing with Dict action space
            using_hybrid_actions = hasattr(env.action_space, "spaces")
            n_agents = env.n_agents

        case EnvironmentEnum.MPE:
            env = simple_spread_v3.env(
                N=3,
                local_ratio=0.5,
                max_cycles=25,
                continuous_actions=True,
                dynamic_rescaling=False,
            )
            state_dim = env.observation_space("agent_0").shape[0]
            action_dim = env.action_space("agent_0").shape[0]
            n_agents = env.max_num_agents

    # Create trainer
    trainer = IPPOTrainer(
        env,
        n_agents,
        state_dim,
        action_dim,
        ppo_config,
        dirs,
        using_hybrid_actions,
        device,
    )

    # trainer.train_normalizer()

    # Train
    trainer.train(
        total_steps=params.n_total_steps,
        batch_size=params.batch_size,
        minibatches=params.n_minibatches,
    )
    trainer.save_training_stats(dirs["logs"] / "training_stats_finished.pkl")

    # Save trained agents
    trainer.save_agents(dirs["models"] / "models_finished.pth")

    # Test trained agents with rendering
    print("\nTesting trained agents...")
    trainer.env.render_mode = "human"
    trainer.render_episode()
    trainer.env.close()
