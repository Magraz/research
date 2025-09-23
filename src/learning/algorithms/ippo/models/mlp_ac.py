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

    def normalize_state(self, state, update=False):
        """
        Use Welford's algorithm to normalize a state, and optionally update the statistics
        for normalizing states using the new state, online.
        """
        state = state.squeeze()

        if self.welford_state_n == 1:
            self.welford_state_mean = torch.zeros(state.size(-1))
            self.welford_state_mean_diff = torch.ones(state.size(-1))

        if update:
            if len(state.size()) == 1:  # if we get a single state vector
                state_old = self.welford_state_mean
                self.welford_state_mean += (state - state_old) / self.welford_state_n
                self.welford_state_mean_diff += (state - state_old) * (
                    state - state_old
                )
                self.welford_state_n += 1
            else:
                raise RuntimeError  # this really should not happen

        return (state - self.welford_state_mean) / torch.sqrt(
            self.welford_state_mean_diff / self.welford_state_n
        )

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

    # def _base_normal(self, mu):
    #     """Unbounded Normal N(mu, diag(std^2)) in R^act_dim."""
    #     log_std = self.log_action_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
    #     std = torch.exp(log_std)
    #     return Independent(Normal(mu, std), 1)  # sums over last dim

    # def _squashed_dist(self, base):
    #     """
    #     TransformedDistribution with tanh, then affine to [low, high].
    #     This handles log_prob (with Jacobians) correctly.
    #     """
    #     # tanh -> [-1,1]
    #     squashed = TransformedDistribution(base, [TanhTransform(cache_size=1)])
    #     # affine map from [-1,1] -> [low, high]
    #     scale = (self.high - self.low) / 2
    #     loc = (self.high + self.low) / 2
    #     return TransformedDistribution(
    #         squashed, [AffineTransform(loc=loc, scale=scale)]
    #     )

    # def get_action_dist(self, mu):
    #     """Full transformed action distribution on [low, high]."""
    #     base = self._base_normal(mu)
    #     return self._squashed_dist(base)

    # def _entropy(self, dist):
    #     """Monte Carlo entropy estimate (exact closed-form is messy after transforms)."""
    #     a = dist.rsample()
    #     return -dist.log_prob(a)

    def act(self, state, update_norm=False, deterministic=False):

        # state = self.normalize_state(state, update_norm)

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

        # state = self.normalize_state(state)

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
