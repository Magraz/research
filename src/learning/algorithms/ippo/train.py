import torch
from learning.algorithms.ippo.trainer import IPPOTrainer
from learning.environments.types import EnvironmentParams
from learning.algorithms.ippo.types import Experiment


def train(
    exp_config: Experiment,
    env_config: EnvironmentParams,
    device: str,
    trial_id: str,
    dirs: dict,
    checkpoint: bool = False,
):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Environment configuration
    env_config = {
        "render_mode": None,  # Set to "human" for visual training
        "n_agents": 10,
    }

    # PPO configuration
    ppo_config = {
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
    }

    # Create trainer
    trainer = IPPOTrainer(env_config, ppo_config, device)

    # Train
    trainer.train(
        num_episodes=2000,
        log_every=50,
    )
    trainer.save_training_stats(dirs["logs"] / "training_stats.pkl")

    # Save trained agents
    trainer.save_agents(dirs["models"] / "models.pth")

    # Test trained agents with rendering
    print("\nTesting trained agents...")
    trainer.env.render_mode = "human"
    trainer.render_episode()
    trainer.env.close()
