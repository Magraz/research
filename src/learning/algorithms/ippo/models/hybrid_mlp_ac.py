import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# To this:
class Hybrid_MLP_AC(nn.Module):
    """Actor-critic network that supports Dict action spaces with movement, link_openness, and detach components."""

    def __init__(
        self,
        observation_space: int,
        action_space: int,
        hidden_dim: int = 128,
    ):
        super(Hybrid_MLP_AC, self).__init__()

        self.action_space = action_space

        # Shared feature extraction
        self.actor_layer1 = nn.Linear(observation_space, hidden_dim)
        self.actor_layer2 = nn.Linear(hidden_dim, hidden_dim)

        # Movement action (continuous 2D)
        movement_dim = action_space["movement"].shape[-1]
        self.movement_mean = nn.Linear(hidden_dim, movement_dim)
        self.movement_log_std = nn.Parameter(torch.ones(1, movement_dim) * -0.5)

        # Link openness action (discrete binary)
        self.link_openness_logits = nn.Linear(hidden_dim, 1)  # Single binary output

        # Detach action (continuous scalar)
        self.detach_mean = nn.Linear(hidden_dim, 1)
        self.detach_log_std = nn.Parameter(torch.ones(1, 1) * -0.5)

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(observation_space, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        # Shared feature extraction
        x = F.gelu(self.actor_layer1(state))
        x = F.gelu(self.actor_layer2(x))

        # Movement action distribution
        movement_mean = torch.tanh(self.movement_mean(x))
        movement_log_std = self.movement_log_std.expand_as(movement_mean)

        # Link openness logits
        link_openness_logits = self.link_openness_logits(x)

        # Detach action distribution
        detach_mean = torch.sigmoid(self.detach_mean(x))  # Bound to [0,1]
        detach_log_std = self.detach_log_std.expand_as(detach_mean)

        value = self.critic(state)

        return {
            "movement": (movement_mean, movement_log_std),
            "link_openness": link_openness_logits,
            "detach": (detach_mean, detach_log_std),
        }, value

    def act(self, state, deterministic=False):
        action_params, value = self.forward(state)

        # Movement action (continuous)
        movement_mean, movement_log_std = action_params["movement"]

        if deterministic:
            movement = movement_mean
        else:
            movement_std = torch.exp(movement_log_std)
            movement = torch.normal(movement_mean, movement_std)

        # Link openness (discrete binary)
        link_logits = action_params["link_openness"]
        link_probs = torch.sigmoid(link_logits)  # Convert to probability

        if deterministic:
            link_openness = (link_probs > 0.5).int()
        else:
            link_openness = torch.bernoulli(link_probs).int()

        # Detach action (continuous in [0,1])
        detach_mean, detach_log_std = action_params["detach"]

        if deterministic:
            detach = detach_mean
        else:
            detach_std = torch.exp(detach_log_std)
            detach = torch.clamp(torch.normal(detach_mean, detach_std), 0.0, 1.0)

        action = {
            "movement": movement,
            "link_openness": link_openness,
            "detach": detach,
        }

        # Calculate log probability
        log_prob = self._get_log_prob(action, action_params)

        return action, log_prob, value

    def evaluate(self, state, action):
        action_params, value = self.forward(state)
        log_prob = self._get_log_prob(action, action_params)
        entropy = self._get_entropy(action_params)
        return log_prob, value, entropy

    def _get_log_prob(self, action, action_params):
        """Calculate combined log probability for all action components"""
        # Movement log prob
        movement_mean, movement_log_std = action_params["movement"]
        movement_std = torch.exp(movement_log_std)
        movement_dist = torch.distributions.Normal(movement_mean, movement_std)
        movement_log_prob = movement_dist.log_prob(action["movement"]).sum(-1)

        # Link openness log prob (binary)
        link_logits = action_params["link_openness"]
        link_dist = torch.distributions.Bernoulli(logits=link_logits.squeeze(-1))
        link_log_prob = link_dist.log_prob(action["link_openness"].float().squeeze(-1))

        # Detach log prob
        detach_mean, detach_log_std = action_params["detach"]
        detach_std = torch.exp(detach_log_std)
        detach_dist = torch.distributions.Normal(detach_mean, detach_std)
        detach_log_prob = detach_dist.log_prob(action["detach"]).sum(-1)

        # Combined log probability
        return movement_log_prob + link_log_prob + detach_log_prob

    def _get_entropy(self, action_params):
        """Calculate combined entropy for all action components"""
        # Movement entropy
        movement_mean, movement_log_std = action_params["movement"]
        movement_std = torch.exp(movement_log_std)
        movement_dist = torch.distributions.Normal(movement_mean, movement_std)
        movement_entropy = movement_dist.entropy().sum(-1)

        # Link openness entropy
        link_logits = action_params["link_openness"]
        link_dist = torch.distributions.Bernoulli(logits=link_logits.squeeze(-1))
        link_entropy = link_dist.entropy()

        # Detach entropy
        detach_mean, detach_log_std = action_params["detach"]
        detach_std = torch.exp(detach_log_std)
        detach_dist = torch.distributions.Normal(detach_mean, detach_std)
        detach_entropy = detach_dist.entropy().sum(-1)

        return movement_entropy + link_entropy + detach_entropy

    def _get_legacy_log_prob(self, action, action_mean, action_log_std):
        """Legacy support for simple Box action spaces"""
        action_std = torch.exp(action_log_std)
        action_log_prob = -0.5 * (
            ((action - action_mean) / (action_std + 1e-8)) ** 2
            + 2 * action_log_std
            + math.log(2 * math.pi)
        )
        return action_log_prob.sum(-1)
