import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from typing import List
from torch_geometric.nn import AttentionalAggregation


class ActorCritic(torch.nn.Module):
    def __init__(
        self,
        n_agents_train: int,
        n_agents_eval: int,
        d_state: int,
        d_action: int,
        device: str,
        hidden_dim=128,
    ):
        super(ActorCritic, self).__init__()

        self.n_agents_eval = n_agents_eval
        self.d_action = d_action
        self.device = device

        self.log_action_std = nn.Parameter(
            torch.ones(
                d_action * n_agents_train,
                requires_grad=True,
                device=device,
            )
            * -0.5
        )

        self.gat1 = GATv2Conv(d_state, hidden_dim, heads=2, concat=True)
        self.gat2 = GATv2Conv(hidden_dim * 2, hidden_dim, heads=1, concat=False)

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, d_action),
        )

        # Critic
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self.att_pool = AttentionalAggregation(
            nn.Sequential(nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, 1))
        )

    def create_chain_graph_batch(self, x_tensor):
        """Convert a batched tensor into a list of chain graphs."""
        graphs = []

        for g in range(x_tensor.size(0)):

            x = x_tensor[g]  # (n_nodes, feat_dim)
            n_nodes = x.size(0)

            # Chain edges: i <-> i+1
            edges = [[i, i + 1] for i in range(n_nodes - 1)]
            edges += [[i + 1, i] for i in range(n_nodes - 1)]
            edge_index = (
                torch.tensor(edges, dtype=torch.long, device=self.device)
                .t()
                .contiguous()
            )  # (2, E)

            graphs.append(Data(x=x, edge_index=edge_index))

        return graphs

    def get_action_and_value(self, state):

        graph_list = self.create_chain_graph_batch(state)

        batched_graph = Batch.from_data_list(graph_list)

        x = self.forward(batched_graph)

        graph_emb = self.att_pool(x, batched_graph.batch)
        value = self.value_head(graph_emb)

        mean = (
            self.actor_head(x)
            .reshape((state.shape[0], self.n_agents_eval, self.d_action))
            .flatten(start_dim=1)
        )

        return mean, value

    def forward(self, batch: Batch):

        x = F.gelu(self.gat1(batch.x, batch.edge_index))
        x = F.layer_norm(x, x.shape[1:])  # Add normalization
        x = F.gelu(self.gat2(x, batch.edge_index))

        return x

    def get_value(self, state: torch.Tensor):
        with torch.no_grad():
            _, value = self.get_action_and_value(state)
            return value

    def get_action_dist(self, action_mean):
        action_std = torch.exp(self.log_action_std)
        return Normal(action_mean, action_std)

    def act(self, state, deterministic=False):

        action_mean, value = self.get_action_and_value(state)

        if deterministic:
            return action_mean.detach()

        dist = self.get_action_dist(action_mean)

        action = dist.sample()
        action_logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        return (
            action.detach(),
            action_logprob.detach(),
            value.detach(),
        )

    def evaluate(self, state, action):

        action_mean, value = self.get_action_and_value(state)

        dist = self.get_action_dist(action_mean)

        action_logprobs = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)
        dist_entropy = torch.sum(dist.entropy(), dim=-1, keepdim=True)

        return action_logprobs, value, dist_entropy


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = ActorCritic(
        n_agents_train=4,
        n_agents_eval=4,
        d_state=18,
        d_action=2 * 4,
        device=device,
    ).to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
