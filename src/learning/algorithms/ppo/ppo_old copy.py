import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# from learning.algorithms.ppo.models.mlp_ac import ActorCritic
from learning.algorithms.ppo.types import Params
from torch.distributions.normal import Normal

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


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
        state_dim = params.state_dim
        action_dim = params.action_dim
        actor_hidden = 64
        critic_hidden = 64
        active_fn = nn.LeakyReLU
        self.device = params.device
        n_agents = params.n_agents

        self.action_dim = action_dim
        self.log_action_std = nn.Parameter(
            torch.rand(action_dim * n_agents, requires_grad=True, device=self.device)
            / 100
            + (-0.8)
        ).to(self.device)

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, actor_hidden),
            active_fn(),
            nn.Linear(actor_hidden, actor_hidden),
            active_fn(),
            nn.Linear(actor_hidden, action_dim * n_agents),
            nn.Tanh(),
        )

        # Apply orthogonal initialization
        self.actor.apply(
            lambda m: orthogonal_init(
                m, gain=torch.nn.init.calculate_gain("leaky_relu")
            )
        )

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, critic_hidden),
            active_fn(),
            nn.Linear(critic_hidden, critic_hidden),
            active_fn(),
            nn.Linear(critic_hidden, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, deterministic=False):

        action_mean = self.actor(state)

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

        action_mean = self.actor(state)

        action_std = torch.exp(self.log_action_std)

        dist = Normal(action_mean, action_std)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, params: Params, n_envs: int = 1):

        self.device = params.device
        self.n_envs = n_envs
        self.n_epochs = params.n_epochs
        self.minibatch_size = params.minibatch_size

        self.gamma = params.gamma
        self.lmbda = params.lmbda
        self.eps_clip = params.eps_clip
        self.grad_clip = params.grad_clip
        self.beta_ent = params.beta_ent

        self.buffer = RolloutBuffer()

        # self.policy = ActorCritic(
        #     d_action=params.action_dim,
        #     d_state=params.state_dim,
        #     n_agents=params.n_agents,
        # ).to(self.device)
        self.policy = ActorCritic(params).to(self.device)

        self.opt_actor = torch.optim.Adam(
            [p for p in self.policy.actor.parameters()] + [self.policy.log_action_std],
            lr=params.lr_actor,
        )

        self.opt_critic = torch.optim.Adam(
            self.policy.critic.parameters(), lr=params.lr_critic
        )

        # self.policy_old = ActorCritic(
        #     d_action=params.action_dim,
        #     d_state=params.state_dim,
        #     n_agents=params.n_agents,
        # ).to(self.device)

        self.policy_old = ActorCritic(params).to(self.device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss_fn = nn.MSELoss()

        self.track = params.log_data

        if self.track:
            self.writer = SummaryWriter(params.log_filename)

    def select_action(self, state):
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach()

    def deterministic_action(self, state):
        with torch.no_grad():
            action = self.policy_old.act(state, deterministic=True)

        return action.detach()

    def add_reward_terminal(self, reward: float, done: bool):
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)

    def gae(
        self, rewards: torch.Tensor, values: torch.Tensor, is_terminals: torch.Tensor
    ):
        values = torch.cat(
            (values, torch.tensor([0.0], device=self.device))
        )  # Bootstrap value
        returns = []
        gae = 0.0

        for i in reversed(range(len(rewards))):
            mask = 1.0 - float(is_terminals[i])  # Convert bool to float
            delta = rewards[i] + self.gamma * values[i + 1] * mask - values[i]
            gae = delta + self.gamma * self.lmbda * mask * gae
            returns.insert(0, gae)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (
            returns.std() + 1e-8
        )  # Normalize advantages

        return returns

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):

            if is_terminal:
                discounted_reward = 0

            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # Convert lists to tensors
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(self.device)
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(self.device)

        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(self.device)
        old_state_values = (
            torch.stack(self.buffer.state_values, dim=0).detach().to(self.device)
        )

        # Calculate advantages
        advantages = self.gae(
            torch.tensor(self.buffer.rewards, dtype=torch.float32).to(self.device),
            torch.tensor(self.buffer.state_values, dtype=torch.float32).to(self.device),
            torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(self.device),
        ).unsqueeze(-1)

        Aloss, Closs, Entropy = [], [], []
        # Optimize policy for K epochs
        for _ in range(self.n_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # Match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # Final loss of clipped objective PPO
            loss_actor = -torch.min(surr1, surr2)

            loss_actor = loss_actor - self.beta_ent * dist_entropy

            loss_actor = loss_actor.mean()
            loss_critic = self.loss_fn(state_values, rewards)

            Entropy.append(dist_entropy.mean().item())
            Aloss.append(loss_actor.item())
            Closs.append(loss_critic.item())

            # Take gradient step
            self.opt_actor.zero_grad()
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.grad_clip
            )
            self.opt_actor.step()

            self.opt_critic.zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.grad_clip
            )
            self.opt_critic.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )


if __name__ == "__main__":
    pass
