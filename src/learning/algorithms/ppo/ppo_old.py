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
        action_std_init=0.6,
    ):

        self.action_std = action_std_init

        self.device = params.device
        self.n_epochs = params.n_epochs
        self.minibatch_size = params.minibatch_size

        self.gamma = params.gamma
        self.lmbda = params.lmbda
        self.eps_clip = params.eps_clip
        self.grad_clip = params.grad_clip
        self.beta_ent = params.beta_ent
        self.n_epochs = params.n_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            params,
            action_std_init,
        ).to(self.device)

        self.opt_actor = torch.optim.Adam(
            [p for p in self.policy.actor.parameters()] + [self.policy.log_action_std],
            lr=params.lr_actor,
        )

        self.opt_critic = torch.optim.Adam(
            self.policy.critic.parameters(), lr=params.lr_critic
        )

        self.policy_old = ActorCritic(params, action_std_init).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().flatten()

    def deterministic_action(self, state):
        with torch.no_grad():
            action = self.policy_old.act(state, deterministic=True)

        return action.detach().flatten()

    def update(self):
        # Monte Carlo estimate of returns
        discounted_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(
            self.device
        )
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-7
        )

        # convert list to tensor
        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0))
            .detach()
            .to(self.device)
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0))
            .detach()
            .to(self.device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
            .detach()
            .to(self.device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(self.device)
        )

        # calculate advantages
        advantages = discounted_rewards.detach() - old_state_values.detach()

        # Create old_states and old_actions dataset
        dataset = RolloutData(
            old_states, old_actions, old_logprobs, advantages, discounted_rewards
        )
        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        # Optimize policy for K epochs
        for _ in range(self.n_epochs):

            for (
                b_old_states,
                b_old_actions,
                b_old_logprobs,
                b_advantages,
                b_rewards,
            ) in loader:

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    b_old_states, b_old_actions
                )

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - b_old_logprobs.detach())

                # Finding Surrogate Loss
                surr1 = ratios * b_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * b_advantages
                )

                # Calculate log_std regularization terms
                log_std_penalty = 1e-4 * torch.sum(self.policy.log_action_std.square())
                entropy_bonus = -self.beta_ent * dist_entropy

                # Calculate actor and critic losses
                actor_loss = -torch.min(surr1, surr2) + log_std_penalty + entropy_bonus
                critic_loss = self.loss_fn(state_values, b_rewards)

                # Take actor gradient step
                self.opt_actor.zero_grad()
                actor_loss.mean().backward()
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
