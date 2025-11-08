import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MAPPOActor(nn.Module):
    """Decentralized actor - each agent has its own or they share one"""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        discrete: bool = False,
    ):
        super(MAPPOActor, self).__init__()

        self.discrete = discrete
        self.action_dim = action_dim

        if not discrete:
            # For continuous actions
            self.log_action_std = nn.Parameter(
                torch.full((action_dim,), -0.5, requires_grad=True)
            )

        # Actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

    def forward(self, obs):
        """Get action logits or means"""
        if self.discrete:
            return self.actor(obs)  # logits
        else:
            return self.actor(obs)  # means

    def get_action_dist(self, action_params):
        """Get action distribution"""
        if self.discrete:
            return Categorical(logits=action_params)
        else:
            log_std = self.log_action_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
            action_std = torch.exp(log_std)
            return Normal(action_params, action_std)

    def act(self, obs, deterministic=False):
        """Sample action from policy"""
        action_params = self.forward(obs)
        dist = self.get_action_dist(action_params)

        if self.discrete:
            if deterministic:
                action = action_params.argmax(dim=-1, keepdim=True)
            else:
                action = dist.sample().unsqueeze(-1)
            logprob = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
        else:
            if deterministic:
                action = action_params
            else:
                action = dist.sample()
            logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        return action, logprob

    def evaluate(self, obs, action):
        """Evaluate actions for training"""
        action_params = self.forward(obs)
        dist = self.get_action_dist(action_params)

        if self.discrete:
            action_squeezed = action.squeeze(-1) if action.dim() > 1 else action
            logprob = dist.log_prob(action_squeezed).unsqueeze(-1)
            entropy = dist.entropy().unsqueeze(-1)
        else:
            logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)
            entropy = torch.sum(dist.entropy(), dim=-1, keepdim=True)

        return logprob, entropy


class MAPPOCritic(nn.Module):
    """Centralized critic - observes global state"""

    def __init__(
        self,
        global_state_dim: int,
        hidden_dim: int = 256,
    ):
        super(MAPPOCritic, self).__init__()

        # Larger network for centralized critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(global_state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, global_state):
        """Get value estimate from global state"""
        return self.critic(global_state)


class MAPPONetwork(nn.Module):
    """Combined MAPPO network with shared/individual actors and centralized critic"""

    def __init__(
        self,
        observation_dim: int,
        global_state_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dim: int = 128,
        discrete: bool = False,
        share_actor: bool = True,  # Whether to share actor parameters
    ):
        super(MAPPONetwork, self).__init__()

        self.n_agents = n_agents
        self.discrete = discrete
        self.share_actor = share_actor

        if share_actor:
            # Single shared actor for all agents
            self.actor = MAPPOActor(observation_dim, action_dim, hidden_dim, discrete)
        else:
            # Separate actor for each agent
            self.actors = nn.ModuleList(
                [
                    MAPPOActor(observation_dim, action_dim, hidden_dim, discrete)
                    for _ in range(n_agents)
                ]
            )

        # Centralized critic (always shared)
        self.critic = MAPPOCritic(global_state_dim, hidden_dim * 2)

    def get_actor(self, agent_idx):
        """Get the actor for a specific agent"""
        if self.share_actor:
            return self.actor
        else:
            return self.actors[agent_idx]

    def act(self, obs, agent_idx, deterministic=False):
        """Get action for a specific agent"""
        actor = self.get_actor(agent_idx)
        return actor.act(obs, deterministic)

    def evaluate_actions(self, obs, global_states, actions, agent_idx):
        """Evaluate actions for training"""
        # Get log probs and entropy from actor
        actor = self.get_actor(agent_idx)
        log_probs, entropy = actor.evaluate(obs, actions)

        # Get values from centralized critic
        values = self.critic(global_states).squeeze(-1)

        return log_probs, values, entropy

    def get_value(self, global_state):
        """Get value from centralized critic"""
        return self.critic(global_state)
