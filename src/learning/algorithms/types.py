from dataclasses import dataclass


class Team:
    def __init__(
        self,
        individuals: list = None,
        combination: list = None,
    ):
        self.individuals = individuals if individuals is not None else []
        self.combination = combination if combination is not None else []
