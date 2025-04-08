import torch
import torch.nn as nn

from torch.distributions.normal import Normal


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
    def __init__(self, dim_action, dim_state):
        super(ActorCritic, self).__init__()

        actor_hidden = 64
        critic_hidden = 64
        active_fn = nn.LeakyReLU

        self.log_action_std = nn.Parameter(torch.zeros(dim_action, requires_grad=True))

        # Actor
        self.actor_policy = nn.Sequential(
            nn.Linear(dim_state, actor_hidden),
            active_fn(),
            nn.Linear(actor_hidden, actor_hidden),
            active_fn(),
            nn.Linear(actor_hidden, dim_action),
            nn.Tanh(),
        )

        # Apply orthogonal initialization
        self.actor_policy.apply(
            lambda m: orthogonal_init(
                m, gain=torch.nn.init.calculate_gain("leaky_relu")
            )
        )

        self.actor = self.actor_policy.parameters()

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(dim_state, critic_hidden),
            active_fn(),
            nn.Linear(critic_hidden, critic_hidden),
            active_fn(),
            nn.Linear(critic_hidden, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, deterministic=False, **kwargs):

        action_mean = self.actor_policy(state)

        if deterministic:
            return action_mean.detach()

        action_std = torch.exp(self.log_action_std)

        dist = Normal(action_mean, action_std)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        state_val = self.critic(state)

        return (
            action.detach(),
            action_logprob.detach(),
            state_val.detach(),
        )

    def evaluate(self, state, action):

        action_mean = self.actor_policy(state)

        action_std = torch.exp(self.log_action_std)

        dist = Normal(action_mean, action_std)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy
