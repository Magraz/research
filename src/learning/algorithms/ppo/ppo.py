import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from learning.algorithms.ppo.models.mlp_ac import ActorCritic
from learning.algorithms.ppo.types import Params


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


class RolloutData(Dataset):
    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        advantages: torch.Tensor,
        rewards: torch.Tensor,
    ):
        self.states = states
        self.actions = actions
        self.logprobs = logprobs
        self.advantages = advantages
        self.rewards = rewards

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.logprobs[idx],
            self.advantages[idx],
            self.rewards[idx],
        )


class PPO:
    def __init__(
        self,
        params: Params,
        n_envs: int = 1,
    ):

        self.device = params.device
        self.n_epochs = params.n_epochs
        self.minibatch_size = params.minibatch_size

        self.n_envs = n_envs
        self.gamma = params.gamma
        self.lmbda = params.lmbda
        self.eps_clip = params.eps_clip
        self.grad_clip = params.grad_clip
        self.ent_coef = params.ent_coef
        self.std_coef = params.std_coef
        self.n_epochs = params.n_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            n_agents=params.n_agents,
            d_state=params.state_dim,
            d_action=params.action_dim,
            device=params.device,
        ).to(self.device)

        self.opt_actor = torch.optim.Adam(
            [p for p in self.policy.actor.parameters()] + [self.policy.log_action_std],
            lr=params.lr_actor,
        )

        self.opt_critic = torch.optim.Adam(
            self.policy.critic.parameters(), lr=params.lr_critic
        )

        self.policy_old = ActorCritic(
            n_agents=params.n_agents,
            d_state=params.state_dim,
            d_action=params.action_dim,
            device=params.device,
        ).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss_fn = nn.MSELoss()

    def select_action(self, state: torch.Tensor):
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
        self,
        bootstrapped_values: list[torch.Tensor],
    ):
        values = torch.stack(bootstrapped_values).squeeze()
        rewards = torch.stack(self.buffer.rewards)
        is_terminals = torch.stack(self.buffer.is_terminals)

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(self.n_envs)

        timesteps, _ = rewards.shape

        for t in reversed(range(timesteps)):
            mask = 1 - is_terminals[t].int()  # Convert bool to float
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.lmbda * mask * gae
            advantages[t] = gae

        returns = advantages + values[:-1, :]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return (
            advantages.transpose(1, 0).flatten().unsqueeze(-1).detach(),
            returns.transpose(1, 0).flatten().unsqueeze(-1).detach(),
        )

    def bootstrap_episodes(self):
        # Bootstrap episodes by calculating the value of the final state if not terminated
        with torch.no_grad():
            final_values = self.policy_old.critic(
                self.buffer.states[-1]
            ) * ~self.buffer.is_terminals[-1].unsqueeze(-1)

        bootstrapped_values = self.buffer.state_values.copy()
        bootstrapped_values.append(final_values)

        # bootstrapped_rewards = self.buffer.rewards.copy()
        # bootstrapped_rewards.append(final_values.squeeze())

        # bootstrapped_is_terminals = self.buffer.is_terminals.copy()
        # bootstrapped_is_terminals.append(torch.ones(self.n_envs, dtype=bool))

        return bootstrapped_values  # , bootstrapped_rewards, bootstrapped_is_terminals

    def get_discounted_rewards(
        self,
        bootstrapped_rewards: list[torch.Tensor],
        bootstrapped_is_terminals: list[torch.Tensor],
    ):
        # Monte Carlo estimate of returns
        discounted_rewards = []
        discounted_reward = torch.zeros(self.n_envs, device=self.device)
        for reward, is_terminal in zip(
            reversed(bootstrapped_rewards),
            reversed(bootstrapped_is_terminals),
        ):
            # If episode terminated reset discounting
            discounted_reward *= ~is_terminal

            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        discounted_rewards = (
            torch.stack(discounted_rewards[:-1])
            .transpose(1, 0)
            .flatten(end_dim=1)
            .unsqueeze(-1)
        )

        # Normalizing the rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-8
        )

        return discounted_rewards.detach()

    def update(self):

        # Bootstrapped incomplete episodes
        boots_values = self.bootstrap_episodes()

        # Calculate advantages
        advantages, returns = self.gae(boots_values)

        # Convert buffer lists from list of (n_env,dim) per timestep to a tensor of shape (timestep*n_envs, dim)
        old_states = (
            torch.stack(self.buffer.states).transpose(1, 0).flatten(end_dim=1).detach()
        )
        old_actions = (
            torch.stack(self.buffer.actions).transpose(1, 0).flatten(end_dim=1).detach()
        )
        old_logprobs = (
            torch.stack(self.buffer.logprobs)
            .transpose(1, 0)
            .flatten(end_dim=1)
            .detach()
        )

        # Create dataset from rollout
        dataset = RolloutData(
            old_states, old_actions, old_logprobs, advantages, returns
        )
        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        # Optimize policy for n epochs
        for _ in range(self.n_epochs):

            for (
                b_old_states,
                b_old_actions,
                b_old_logprobs,
                b_advantages,
                b_returns,
            ) in loader:

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    b_old_states, b_old_actions
                )

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - b_old_logprobs)

                # Finding Surrogate Loss
                surr1 = ratios * b_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * b_advantages
                )

                # Penalize high values of log_std by increasing the loss, thus decreasing exploration
                log_std_penalty = (
                    self.std_coef * self.policy.log_action_std.square().mean()
                )

                # Promote exploration by reducing the loss if entropy increases
                entropy_bonus = self.ent_coef * dist_entropy.mean()

                ppo_loss = -torch.min(surr1, surr2).mean()

                # Calculate actor and critic losses
                actor_loss = ppo_loss + log_std_penalty - entropy_bonus
                critic_loss = self.loss_fn(state_values, b_returns)

                # Take actor gradient step
                self.opt_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), self.grad_clip
                )
                self.opt_actor.step()

                # Take critic gradient step
                self.opt_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(), self.grad_clip
                )
                self.opt_critic.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
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
