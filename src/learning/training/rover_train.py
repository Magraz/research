from deap import base
from deap import creator
from deap import tools
import torch
import torch.nn.functional as F

import random

from vmas.simulator.environment import Environment
from vmas.simulator.utils import save_video

from learning.algorithms.ccea.policies.mlp import MLP_Policy
from learning.algorithms.ccea.policies.gru import GRU_Policy

from learning.environments.create_env import create_env
from learning.algorithms.ccea.selection import (
    binarySelection,
    epsilonGreedySelection,
    softmaxSelection,
)
from learning.algorithms.ccea.types import (
    EvalInfo,
    PolicyEnum,
    SelectionEnum,
    FitnessShapingEnum,
    InitializationEnum,
    FitnessCalculationEnum,
)
from learning.algorithms.ccea.types import CCEA_Config, CCEA_PolicyConfig, Team
from learning.environments.types import EnvironmentEnum

from copy import deepcopy
import numpy as np
import os
from pathlib import Path
import logging
import pickle
import csv

from itertools import combinations

# Create and configure logger
logging.basicConfig(format="%(asctime)s %(message)s")

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


class RoverTrain:
    def __init__(
        self,
        batch_dir: str,
        trials_dir: str,
        trial_id: int,
        trial_name: str,
        video_name: str,
        device: str,
        ccea_config: CCEA_Config,
        **kwargs,
    ):
        ccea_config = CCEA_Config(**ccea_config)
        policy_config = CCEA_PolicyConfig(**ccea_config.policy_config)

        self.batch_dir = batch_dir
        self.trials_dir = trials_dir
        self.trial_name = trial_name
        self.trial_id = trial_id
        self.video_name = video_name

        # Flags
        self.use_teaming = kwargs.pop("use_teaming", False)

        # Environment data
        self.device = device
        self.environment = kwargs.pop("environment", None)
        self.map_size = kwargs.pop("map_size", [])
        self.observation_size = kwargs.pop("observation_size", 0)
        self.action_size = kwargs.pop("action_size", 0)
        self.n_agents = kwargs.pop("n_agents", 0)
        self.n_pois = kwargs.pop("n_pois", 0)
        self.team_size = (
            kwargs.pop("team_size", 0) if self.use_teaming else self.n_agents
        )

        # Experiment Data
        self.n_gens_between_save = kwargs.pop("n_gens_between_save", 0)

        # Policy
        self.output_multiplier = policy_config.output_multiplier
        self.policy_hidden_layers = policy_config.hidden_layers
        self.policy_type = policy_config.type
        self.weight_initialization = policy_config.weight_initialization

        # CCEA
        self.n_gens = ccea_config.n_gens
        self.n_steps = ccea_config.n_steps
        self.subpop_size = ccea_config.subpopulation_size
        self.n_mutants = self.subpop_size // 2
        self.selection_method = ccea_config.selection
        self.fitness_shaping_method = ccea_config.fitness_shaping
        self.fitness_calculation = ccea_config.fitness_calculation
        self.max_std_dev = ccea_config.mutation["max_std_deviation"]
        self.min_std_dev = ccea_config.mutation["min_std_deviation"]
        self.mutation_mean = ccea_config.mutation["mean"]

        # Create the type of fitness we're optimizing
        creator.create("Individual", np.ndarray, fitness=0.0)

        # Now set up the toolbox
        self.toolbox = base.Toolbox()

        self.toolbox.register(
            "subpopulation",
            tools.initRepeat,
            list,
            self.createIndividual,
            n=self.subpop_size,
        )

        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.subpopulation,
            n=self.n_agents,
        )
