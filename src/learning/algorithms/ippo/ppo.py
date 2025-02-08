import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


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


class ActorCritic(nn.Module):
    def __init__(self, params):
        super(ActorCritic, self).__init__()
        state_dim = params.state_dim
        action_dim = params.action_dim
        actor_hidden = params.actor_hidden
        critic_hidden = params.critic_hidden
        active_fn = params.active_fn
        self.device = params.device

        self.action_dim = action_dim
        self.log_action_std = nn.Parameter(
            torch.rand(action_dim, requires_grad=True, device=self.device) / 100
            + params.action_std
        ).to(self.device)

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, actor_hidden),
            active_fn(),
            nn.Linear(actor_hidden, actor_hidden),
            active_fn(),
            nn.Linear(actor_hidden, action_dim),
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
    def __init__(self, params, idx=0, n_buffers=1):

        self.params = params
        self.device = params.device

        self.gamma = params.gamma
        self.lmbda = params.lmbda
        self.eps_clip = params.eps_clip
        self.K_epochs = params.K_epochs

        self.buffers = [RolloutBuffer() for i in range(n_buffers)]

        self.policy = ActorCritic(params).to(self.device)
        self.opt_actor = torch.optim.Adam(
            [p for p in self.policy.actor.parameters()] + [self.policy.log_action_std],
            lr=params.lr_actor,
        )
        self.opt_critic = torch.optim.Adam(
            self.policy.critic.parameters(), lr=params.lr_critic
        )

        self.policy_old = ActorCritic(params).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MSELoss = nn.MSELoss()

    def select_action(self, state, n_buffer):
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffers[n_buffer].states.append(state.squeeze(0))
        self.buffers[n_buffer].actions.append(action.squeeze(0))
        self.buffers[n_buffer].logprobs.append(action_logprob.squeeze(0))
        self.buffers[n_buffer].state_values.append(state_val.squeeze(0))

        return action.detach().flatten()

    def deterministic_action(self, state):
        with torch.no_grad():
            action = self.policy_old.act(state, deterministic=True)

        return action.detach().flatten()

    def add_reward_terminal(self, reward: float, done: bool, n_buffer: int):
        self.buffers[n_buffer].rewards.append(reward)
        self.buffers[n_buffer].is_terminals.append(done)

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
        for buffer in self.buffers:

            rewards = []
            discounted_reward = 0

            for reward, is_terminal in zip(
                reversed(buffer.rewards), reversed(buffer.is_terminals)
            ):

                if is_terminal:
                    discounted_reward = 0

                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            # Normalizing the rewards
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

            # Convert lists to tensors
            old_states = torch.stack(buffer.states, dim=0).detach().to(self.device)
            old_actions = torch.stack(buffer.actions, dim=0).detach().to(self.device)

            old_logprobs = torch.stack(buffer.logprobs, dim=0).detach().to(self.device)
            old_state_values = (
                torch.stack(buffer.state_values, dim=0).detach().to(self.device)
            )

            # Calculate advantages
            advantages = self.gae(
                torch.tensor(buffer.rewards, dtype=torch.float32).to(self.device),
                torch.tensor(buffer.state_values, dtype=torch.float32).to(self.device),
                torch.tensor(buffer.is_terminals, dtype=torch.float32).to(self.device),
            ).unsqueeze(-1)

            Aloss, Closs, Entropy = [], [], []
            # Optimize policy for K epochs
            for _ in range(self.K_epochs):

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
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * advantages
                )

                # Final loss of clipped objective PPO
                loss_actor = -torch.min(surr1, surr2)

                loss_actor = loss_actor - self.params.beta_ent * dist_entropy

                loss_actor = loss_actor.mean()
                loss_critic = self.MSELoss(state_values, rewards)

                Entropy.append(dist_entropy.mean().item())
                Aloss.append(loss_actor.item())
                Closs.append(loss_critic.item())

                # Take gradient step
                self.opt_actor.zero_grad()
                loss_actor.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), self.params.grad_clip
                )
                self.opt_actor.step()

                self.opt_critic.zero_grad()
                loss_critic.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(), self.params.grad_clip
                )
                self.opt_critic.step()

        # if self.params.log_indiv:
        #     prefix = "Agent" + str(self.idx) + "/"
        #     self.params.writer.add_scalar(
        #         prefix + "Loss/entropy", np.mean(Entropy), idx
        #     )
        #     self.params.writer.add_scalar(prefix + "Loss/actor", np.mean(Aloss), idx)
        #     self.params.writer.add_scalar(prefix + "Loss/critic", np.mean(Closs), idx)
        #     self.params.writer.add_scalar(
        #         prefix + "Action/STD_Mean", torch.mean(self.policy.log_action_std), idx
        #     )
        #     # elf.params.writer.add_scalars(prefix+"Action/STD_Vals",{str(i):self.policy.log_action_var[i] for i in range(self.params.action_dim)},idx)
        #     self.params.writer.add_scalar(
        #         prefix + "Loss/Advantage_min", advantages.min().item(), idx
        #     )
        #     self.params.writer.add_scalar(
        #         prefix + "Loss/Advantage_max", advantages.max().item(), idx
        #     )

        # self.params.writer.add_scalar(
        #     prefix + "Reward", sum(self.buffer.rewards) / self.params.N_batch, idx
        # )

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        for buffer in self.buffers:
            buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )


class Params:
    def __init__(self, fname=None, n_agents=0):
        self.K_epochs = 20  # update policy for K epochs in one PPO update
        self.N_batch = 8
        self.N_steps = 3e6
        self.eps_clip = 0.2  # clip parameter for PPO
        self.gamma = 0.99  # discount factor

        self.lr_actor = 0.0003  # learning rate for actor network
        self.lr_critic = 0.001  # learning rate for critic network
        self.action_std = -0.8
        self.random_seed = 0
        self.grad_clip = 1.0

        self.action_dim = 4
        self.state_dim = 24

        self.actor_hidden = 64
        self.critic_hidden = 64
        self.active_fn = nn.LeakyReLU

        self.lmbda = 0.95

        self.device = "cpu"
        self.log_indiv = True

        if fname is not None:
            self.writer = SummaryWriter(fname)
        else:
            self.log_indiv = False

        self.beta_ent = 0.001
        self.n_agents = n_agents

    def write(self):
        for key, val in self.__dict__.items():
            self.writer.add_text("Params/" + key, key + " : " + str(val))


if __name__ == "__main__":
    pass
