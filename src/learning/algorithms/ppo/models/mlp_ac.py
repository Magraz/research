import torch
import torch.nn as nn

from torch.distributions import Normal
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
    def __init__(self, params: Params):
        super(ActorCritic, self).__init__()

        self.device = params.device

        self.log_action_std = nn.Parameter(
            torch.zeros(
                params.action_dim * params.n_agents,
                requires_grad=True,
                device=params.device,
            )
        ).to(self.device)

        # actor
        self.actor = nn.Sequential(
            nn.Linear(params.state_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, params.action_dim * params.n_agents),
            nn.Tanh(),
        ).to(self.device)

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
        ).to(self.device)

    def forward(self):
        raise NotImplementedError

    def get_action_dist(self, action_mean):
        action_std = torch.exp(self.log_action_std)
        return Normal(action_mean, action_std)

    def act(self, state, deterministic=False):

        action_mean = self.actor(state)

        if deterministic:
            return action_mean.detach()

        dist = self.get_action_dist(action_mean)

        action = dist.sample()
        action_logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        state_val = self.critic(state)

        return (
            action.detach(),
            action_logprob.detach(),
            state_val.detach(),
        )

    def evaluate(self, state, action):

        action_mean = self.actor(state)

        dist = self.get_action_dist(action_mean)

        action_logprobs = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)
        dist_entropy = torch.sum(dist.entropy(), dim=-1, keepdim=True)

        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy
