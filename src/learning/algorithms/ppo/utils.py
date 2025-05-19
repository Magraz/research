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
        ):
            match (state_representation):
                case "local":
                    return obs_shape
