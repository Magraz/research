from learning.environments.types import EnvironmentParams, EnvironmentEnum
from learning.algorithms.ippo.types import Experiment, Params
from learning.algorithms.runner import Runner
from pathlib import Path

from learning.algorithms.ippo.trainer import IPPOTrainer

import torch
import numpy as np
import random


def set_seeds(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed_all(seed)  # PyTorch CUDA
    # torch.backends.cudnn.deterministic = True  # Make CUDA deterministic
    # torch.backends.cudnn.benchmark = False  # Disable CUDA benchmarking


class IPPO_Runner(Runner):
    def __init__(
        self,
        device: str,
        batch_dir: Path,
        trials_dir: Path,
        trial_id: str,
        checkpoint: bool,
        exp_config: Experiment,
        env_config: EnvironmentParams,
    ):
        super().__init__(device, batch_dir, trials_dir, trial_id, checkpoint)

        self.exp_config = exp_config
        self.env_config = env_config

        # Set params
        self.params = Params(**self.exp_config.params)

        # Set seeds
        random_seed = self.params.random_seeds[0]

        if self.trial_id.isdigit():
            random_seed = self.params.random_seeds[int(self.trial_id)]

        # Set all random seeds for reproducibility
        set_seeds(random_seed)

        # Device configuration
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.exp_config.device}")

        # Create environment
        n_agents = 0

        match (self.env_config.environment):
            case EnvironmentEnum.BOX2D_SALP:
                from learning.environments.box2d_salp.domain import SalpChainEnv

                # Environment configuration
                self.env = SalpChainEnv(n_agents=env_config.n_agents)
                state_dim = self.env.observation_space.shape[1]
                action_dim = 4
                n_agents = self.env.n_agents

            case EnvironmentEnum.MPE_SPREAD:
                from mpe2 import simple_spread_v3

                self.env = simple_spread_v3.parallel_env(
                    N=3,
                    local_ratio=0.5,
                    max_cycles=25,
                    continuous_actions=False,
                    dynamic_rescaling=True,
                )
                state_dim = self.env.observation_space("agent_0").shape[0]
                action_dim = self.env.action_space("agent_0").n
                n_agents = self.env.max_num_agents

            case EnvironmentEnum.MPE_SIMPLE:
                from mpe2 import simple_v3

                self.env = simple_v3.parallel_env(
                    max_cycles=25,
                    continuous_actions=False,
                )
                state_dim = self.env.observation_space("agent_0").shape[0]
                action_dim = self.env.action_space("agent_0").n
                n_agents = self.env.max_num_agents

        # Create trainer
        self.trainer = IPPOTrainer(
            self.env,
            self.env_config.environment,
            n_agents,
            state_dim,
            action_dim,
            self.params,
            self.dirs,
            self.device,
        )

    def train(self):
        # Train
        self.trainer.train(
            total_steps=self.params.n_total_steps,
            batch_size=self.params.batch_size,
            minibatches=self.params.n_minibatches,
            epochs=self.params.n_epochs,
        )
        self.trainer.save_training_stats(
            self.dirs["logs"] / "training_stats_finished.pkl"
        )

        # Save trained agents
        self.trainer.save_agents(self.dirs["models"] / "models_finished.pth")

        self.trainer.env.close()

    def view(self):
        if self.env_config.environment == EnvironmentEnum.MPE_SPREAD:
            from mpe2 import simple_spread_v3

            self.env = simple_spread_v3.parallel_env(
                N=3,
                local_ratio=0.5,
                max_cycles=25,
                continuous_actions=False,
                dynamic_rescaling=True,
                render_mode="human",
            )
            self.trainer.env = self.env

        elif self.env_config.environment == EnvironmentEnum.MPE_SIMPLE:
            from mpe2 import simple_v3

            self.env = simple_v3.parallel_env(
                max_cycles=25,
                render_mode="human",
            )
            self.trainer.env = self.env

        else:
            self.env.render_mode = "human"

        # Save trained agents
        self.trainer.load_agents(self.dirs["models"] / "models_checkpoint.pth")

        # Test trained agents with rendering
        print("\nTesting trained agents...")
        for i in range(10):
            self.trainer.render_episode(max_steps=512)

        self.trainer.env.close()

    def evaluate(self):
        pass
