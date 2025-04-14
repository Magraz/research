import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal
from learning.algorithms.ppo.types import Params


def orthogonal_init(m, gain=1.0):
    """
    Applies orthogonal initialization to the model layers.

    Parameters:
        m (torch.nn.Module): The layer to initialize.
        gain (float): Scaling factor for the weights.
                      - Use 1.0 for standard layers.
                      - Use sqrt(2) for ReLU activations.
                      - Use 0.01 for small initial weights.
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)  # Apply orthogonal init
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # Set biases to zero


class ActorCritic(nn.Module):
    def __init__(self, params: Params, action_std_init):
        super(ActorCritic, self).__init__()

        self.device = params.device

        self.log_action_std = nn.Parameter(
            torch.zeros(params.action_dim * params.n_agents, requires_grad=True)
        ).to(self.device)

        # actor
        self.actor = nn.Sequential(
            nn.Linear(params.state_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, params.action_dim * params.n_agents),
            nn.Tanh(),
        )

        # Apply orthogonal initialization
        self.actor.apply(
            lambda m: orthogonal_init(
                m, gain=torch.nn.init.calculate_gain("leaky_relu")
            )
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(params.state_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(
        self,
        state,
        deterministic=False,
    ):

        action_mean = self.actor(state)

        if deterministic:
            return action_mean.detach()

        action_std = torch.exp(self.log_action_std)
        action_var = action_std.square()

        cov_mat = torch.diag(action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        action_mean = self.actor(state)

        action_std = torch.exp(self.log_action_std)
        action_var = action_std.square()

        action_var = action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy
