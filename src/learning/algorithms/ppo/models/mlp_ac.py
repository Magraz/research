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
    def __init__(self, n_agents: int, d_state: int, d_action: int, device: str):
        super(ActorCritic, self).__init__()

        self.log_action_std = nn.Parameter(
            torch.zeros(
                d_action * n_agents,
                requires_grad=True,
                device=device,
            )
        )

        # actor
        self.actor = nn.Sequential(
            nn.Linear(d_state, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, d_action * n_agents),
            nn.Tanh(),
        )

        self.actor_params = list(self.actor.parameters())

        # Apply orthogonal initialization
        self.actor.apply(
            lambda m: orthogonal_init(
                m, gain=torch.nn.init.calculate_gain("leaky_relu")
            )
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(d_state, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ActorCritic(
        n_agents=8,
        d_state=32,
        d_action=2 * 8,
        device=device,
    ).to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
