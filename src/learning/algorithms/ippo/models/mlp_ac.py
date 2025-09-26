import torch
import torch.nn as nn

from torch.distributions import Normal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform
import numpy as np

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0  # clamp for stability


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP_AC(nn.Module):
    def __init__(
        self,
        observation_space: int,
        action_dim: int,
        hidden_dim: int = 64,
    ):
        super(MLP_AC, self).__init__()

        self.log_action_std = nn.Parameter(
            torch.full((action_dim,), -0.5, requires_grad=True)
        )

        # Actor
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_space, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

        # Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_space, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        # Observation normalization using welford's method
        self.welford_state_mean = torch.zeros(1)
        self.welford_state_mean_diff = torch.ones(1)
        self.welford_state_n = 1

    def forward(self, state):
        action_mean = self.actor(state)
        value = self.critic(state)
        return value, action_mean

    def get_value(self, state: torch.Tensor):
        return self.critic(state)

    def get_action_dist(self, action_mean):
        log_std = self.log_action_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        action_std = torch.exp(log_std)
        return Normal(action_mean, action_std)

    def act(self, state, deterministic=False):

        value, action_mean = self.forward(state)

        dist = self.get_action_dist(action_mean)
        action = dist.sample()
        logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        if deterministic:
            return (
                action_mean,
                logprob,
                value,
            )
        else:
            return (
                action,
                logprob,
                value,
            )

    def evaluate(self, state, action):

        value, action_mean = self.forward(state)

        dist = self.get_action_dist(action_mean)
        logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)
        entropy = torch.sum(dist.entropy(), dim=-1, keepdim=True)

        return logprob, value, entropy


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MLP_AC(
        n_agents_train=4,
        n_agents_eval=4,
        d_state=4 * 18,
        d_action=2 * 4,
        device=device,
    ).to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
