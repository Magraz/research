from learning.environments.types import EnvironmentParams
from learning.algorithms.ppo.types import Experiment
from learning.algorithms.ippo.train import train
from learning.algorithms.ippo.view import view
from learning.algorithms.ppo.evaluate import evaluate
from learning.algorithms.runner import Runner
from pathlib import Path


class IPPO_Runner(Runner):
    def __init__(
        self,
        device: str,
        batch_dir: Path,
        trials_dir: Path,
        trial_id: str,
        checkpoint: bool,
    ):
        super().__init__(device, batch_dir, trials_dir, trial_id, checkpoint)

    def train(
        self,
        exp_config: Experiment,
        env_config: EnvironmentParams,
    ):
        train(
            exp_config,
            env_config,
            self.device,
            self.trial_id,
            self.dirs,
            self.checkpoint,
        )

    def view(self, exp_config: Experiment, env_config: EnvironmentParams):
        view(exp_config, env_config, self.device, self.dirs)

    def evaluate(self, exp_config: Experiment, env_config: EnvironmentParams):
        evaluate(
            exp_config,
            env_config,
            self.device,
            self.trial_id,
            self.dirs,
        )
