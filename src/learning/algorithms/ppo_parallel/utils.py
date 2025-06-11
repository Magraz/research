import torch


def process_state(
    state: list,
    representation: str,
    model: str,
):
    match (model):
        case "mlp":
            match (representation):
                case "global":
                    return state[0]

                case "local":
                    state = torch.stack(state).transpose(1, 0).flatten(start_dim=1)

                    return state

        case (
            "transformer_encoder"
            | "transformer_decoder"
            | "transformer"
            | "transformer_full"
            | "gat"
            | "gcn"
            | "graph_transformer"
        ):
            match (representation):
                case "local":
                    state = torch.stack(state).transpose(1, 0)
                    return state

    return state


def get_state_dim(obs_shape, state_representation: str, model: str, n_agents: int):

    match (model):
        case "mlp":
            match (state_representation):
                case "global":
                    return obs_shape

                case "local":
                    return obs_shape * n_agents

        case (
            "transformer_encoder"
            | "transformer_decoder"
            | "transformer"
            | "transformer_full"
            | "gat"
            | "gcn"
            | "graph_transformer"
        ):
            match (state_representation):
                case "local":
                    return obs_shape


def create_vmas_env_creator(env_config, base_dir, device):
    """Creates a function that instantiates VMAS environments."""
    from learning.environments.create_env import create_env

    def env_creator(n_envs, base_seed=0):
        """Creates a VMAS environment with n_envs parallel environments."""
        return create_env(
            base_dir,
            n_envs,
            n_agents=env_config.n_agents,
            device=device,
            env_name=env_config.environment,
            seed=(
                env_config.seed + base_seed
                if hasattr(env_config, "seed")
                else base_seed
            ),
        )

    return env_creator
