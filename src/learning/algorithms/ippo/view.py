import torch
from learning.algorithms.ippo.trainer import IPPOTrainer
from learning.environments.types import EnvironmentParams, EnvironmentEnum
from learning.algorithms.ippo.types import Experiment
from learning.algorithms.ippo.types import Experiment, Params
from learning.environments.box2d_salp.domain import SalpChainEnv


def view(
    exp_config: Experiment,
    env_config: EnvironmentParams,
    device: str,
    dirs: dict,
):
    # Set params
    params = Params(**exp_config.params)

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
    match (env_config.environment):
        case EnvironmentEnum.BOX2D_SALP:
            # Environment configuration
            env_config = {
                "render_mode": "human",  # Set to "human" for visual training
                "n_agents": env_config.n_agents,
            }
            env = SalpChainEnv(**env_config)
            state_dim = env.observation_space.shape[1]
            n_agents = env.n_agents

    # Create trainer
    trainer = IPPOTrainer(env, n_agents, state_dim, ppo_config, dirs, device)

    # Save trained agents
    trainer.load_agents(dirs["models"] / "models_checkpoint.pth")

    # Test trained agents with rendering
    print("\nTesting trained agents...")
    for i in range(10):
        trainer.render_episode()
    trainer.env.close()
