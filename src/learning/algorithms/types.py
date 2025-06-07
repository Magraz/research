from enum import StrEnum


class AlgorithmEnum(StrEnum):
    CCEA = "ccea"
    IPPO = "ippo"
    PPO = "ppo"
    PPO_PARALLEL = "ppo_parallel"
    TD3 = "td3"
    NONE = "none"
