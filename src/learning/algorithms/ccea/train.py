import csv
import os
import pickle
from learning.algorithms.ccea.ccea import CooperativeCoevolutionaryAlgorithm
from learning.algorithms.ccea.types import Checkpoint, CCEA_ExperimentConfig
from learning.environments.types import EnvironmentConfig
from learning.environments.create_env import create_env

from pathlib import Path


class CCEA_Trainer:
    def __init__(
        self,
        device: str,
        batch_dir: str,
        trials_dir: str,
        trial_id: int,
        trial_name: str,
        video_name: str,
    ):
        self.device = device
        self.batch_dir = batch_dir
        self.trials_dir = trials_dir
        self.trial_name = trial_name
        self.trial_id = trial_id
        self.video_name = video_name
        self.trial_folder_name = "_".join(("trial", str(self.trial_id)))
        self.trial_dir = os.path.join(self.trials_dir, self.trial_folder_name)
        self.checkpoint = Checkpoint()

    def train(
        self,
        env_config: EnvironmentConfig,
        exp_config: CCEA_ExperimentConfig,
    ):
        # Set trial directory name
        fitness_dir = f"{self.trial_dir}/fitness.csv"
        checkpoint_name = os.path.join(self.trial_dir, "checkpoint.pickle")

        # Create directory for saving data
        if not os.path.isdir(self.trial_dir):
            os.makedirs(self.trial_dir)

        if Path(checkpoint_name).is_file():
            self.checkpoint = self.load_checkpoint(
                checkpoint_name, fitness_dir, self.trial_dir
            )

        else:
            # Create csv file for saving evaluation fitnesses
            self.create_log_file(fitness_dir)

        ccea = CooperativeCoevolutionaryAlgorithm(
            self.device,
            env_config,
            exp_config,
        )

        # Load checkpoint
        checkpoint_gen = 0
        pop = None

        if self.checkpoint.exists:

            pop = self.checkpoint.population
            checkpoint_gen = self.checkpoint.generation

        else:
            # Initialize the population
            pop = ccea.toolbox.population()

        # Create environment
        env = create_env(
            self.batch_dir,
            n_envs=ccea.subpop_size,
            env_name=exp_config.environment,
            device=self.device,
        )

        # Train
        for n_gen in range(ccea.n_gens + 1):

            # Get loading bar up to checkpoint
            if self.checkpoint.exists and n_gen <= checkpoint_gen:
                continue

            pop, team, team_fitness, avg_team_fitness = ccea.run(n_gen, pop, env)

            self.write_log_file(fitness_dir, n_gen, avg_team_fitness, team_fitness)

    def create_log_file(self, fitness_dir):
        header = ["gen", "avg_team_fitness", "best_team_fitness"]

        with open(fitness_dir, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows([header])

    def write_log_file(self, fitness_dir, gen, avg_fitness, best_fitness):

        # Now save it all to the csv
        with open(fitness_dir, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([gen, avg_fitness, best_fitness])

    def save_checkpoint(self, checkpoint_dict):
        # Save checkpoint
        with open(os.path.join(self.trial_dir, "checkpoint.pickle"), "wb") as handle:
            pickle.dump(
                checkpoint_dict,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load_checkpoint(
        self,
        checkpoint_name: str,
        fitness_dir: str,
        trial_dir: str,
    ):
        # Load checkpoint file
        with open(checkpoint_name, "rb") as handle:
            checkpoint = pickle.load(handle)
            pop = checkpoint["population"]
            checkpoint_gen = checkpoint["gen"]

        # Set fitness csv file to checkpoint
        new_fit_path = os.path.join(trial_dir, "fitness_edit.csv")
        with open(fitness_dir, "r") as inp, open(new_fit_path, "w") as out:
            writer = csv.writer(out)
            for row in csv.reader(inp):
                if row[0].isdigit():
                    gen = int(row[0])
                    if gen <= checkpoint_gen:
                        writer.writerow(row)
                else:
                    writer.writerow(row)

        # Remove old fitness file
        os.remove(fitness_dir)
        # Rename new fitness file
        os.rename(new_fit_path, fitness_dir)

        checkpoint = Checkpoint(exists=True, population=pop, generation=checkpoint_gen)

        return checkpoint
