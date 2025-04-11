import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import numpy as np

from learning.algorithms.ppo.types import Params

from learning.algorithms.ppo.models.mlp_ac import ActorCritic

# from learning.algorithms.ppo.models.transformer_ac import ActorCritic

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
    def __init__(self, params: Params, n_envs: int = 1, n_buffers: int = 1):

        self.device = params.device
        self.n_envs = n_envs
        self.n_epochs = params.n_epochs
        self.minibatch_size = params.minibatch_size

        self.gamma = params.gamma
        self.lmbda = params.lmbda
        self.eps_clip = params.eps_clip
        self.grad_clip = params.grad_clip
        self.beta_ent = params.beta_ent

        self.buffers = [RolloutBuffer() for i in range(n_buffers)]

        self.policy = ActorCritic(
            d_action=params.action_dim,
            d_state=params.state_dim,
            n_agents=params.n_agents,
        ).to(self.device)

        self.opt_actor = torch.optim.Adam(
            self.policy.actor_params + [self.policy.log_action_std],
            lr=params.lr_actor,
        )

        self.opt_critic = torch.optim.Adam(
            self.policy.critic.parameters(), lr=params.lr_critic
        )

        self.policy_old = ActorCritic(
            d_action=params.action_dim,
            d_state=params.state_dim,
            n_agents=params.n_agents,
        ).to(self.device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss_fn = nn.MSELoss()

        self.track = params.log_data

        if self.track:
            self.writer = SummaryWriter(params.log_filename)

    def select_action(self, state, n_buffer: int = 0):
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(
                state, actions_buffer=self.buffers[n_buffer].actions
            )

        self.buffers[n_buffer].states.append(state)
        self.buffers[n_buffer].actions.append(action)
        self.buffers[n_buffer].logprobs.append(action_logprob)
        self.buffers[n_buffer].state_values.append(state_val)

        return action.detach()

    def deterministic_action(self, state):
        with torch.no_grad():
            action = self.policy_old.act(state, deterministic=True)

        return action.detach().flatten()

    def add_reward_terminal(self, reward: float, done: bool, n_buffer: int = 0):
        self.buffers[n_buffer].rewards.append(reward)
        self.buffers[n_buffer].is_terminals.append(done)

    def gae(
        self, rewards: torch.Tensor, values: torch.Tensor, is_terminals: torch.Tensor
    ):
        values = torch.cat(
            (values, torch.zeros((1, self.n_envs, 1), device=self.device))
        )  # Bootstrap value

        advantages = torch.zeros_like(rewards)
        gae = 0.0

        for i in reversed(range(len(rewards))):
            mask = 1 - float(is_terminals[i])  # Convert bool to float
            delta = (
                rewards[i]
                + self.gamma * values[i + 1].view(self.n_envs) * mask
                - values[i].view(self.n_envs)
            )
            gae = delta + self.gamma * self.lmbda * mask * gae
            advantages[i] = gae

        norm_advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-7
        )  # Normalize advantages
        return norm_advantages

    def update(self):
        # Monte Carlo estimate of returns
        for buffer in self.buffers:

            rewards = []
            discounted_reward = torch.zeros(self.n_envs, device=self.device)

            for reward, is_terminal in zip(
                reversed(buffer.rewards), reversed(buffer.is_terminals)
            ):

                discounted_reward *= ~is_terminal

                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            # Normalizing the rewards
            # rewards = torch.stack(rewards).T.unsqueeze(-1)
            # rewards = rewards.reshape(-1, rewards.shape[-1])

            rewards = torch.stack(rewards).unsqueeze(-1)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
            rewards = rewards.reshape(-1, rewards.shape[-1])

            # Convert lists to tensors
            old_states = torch.stack(buffer.states, dim=0).detach().to(self.device)
            old_states = old_states.reshape(-1, old_states.shape[-1])

            old_actions = torch.stack(buffer.actions, dim=0).detach().to(self.device)
            old_actions = old_actions.reshape(-1, old_actions.shape[-1])

            old_logprobs = (
                torch.stack(buffer.logprobs, dim=0)
                .detach()
                .to(self.device)
                .unsqueeze(-1)
            )
            old_logprobs = old_logprobs.reshape(-1, old_logprobs.shape[-1])

            old_state_values = (
                torch.stack(buffer.state_values, dim=0).detach().to(self.device)
            )
            old_state_values = old_state_values.reshape(-1, old_state_values.shape[-1])

            # Calculate advantages
            advantages = (
                self.gae(
                    torch.stack(buffer.rewards),
                    torch.stack(buffer.state_values),
                    torch.stack(buffer.is_terminals),
                )
                .reshape(-1)
                .unsqueeze(-1)
            )

            # calculate advantages
            # advantages = rewards.detach() - old_state_values.detach()

            actor_loss, critic_loss, entropy = [], [], []

            # Create old_states and old_actions dataset
            dataset = RolloutData(
                old_states, old_actions, old_logprobs, advantages, rewards
            )
            loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

            # Optimize policy for n_epochs
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

                    # Finding the ratio (pi_theta / pi_theta__old)
                    ratios = torch.exp(logprobs - b_old_logprobs)

                    # Finding Surrogate Loss
                    surr1 = ratios * b_advantages
                    surr2 = (
                        torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                        * b_advantages
                    )

                    # Final loss of clipped objective PPO
                    loss_actor = -torch.min(surr1, surr2)

                    loss_actor = loss_actor - self.beta_ent * dist_entropy

                    loss_actor = loss_actor.mean()
                    loss_critic = self.loss_fn(state_values, b_rewards)

                    entropy.append(dist_entropy.mean().item())
                    actor_loss.append(loss_actor.item())
                    critic_loss.append(loss_critic.item())

                    # Take gradient step
                    self.opt_actor.zero_grad()
                    loss_actor.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.actor_params, self.grad_clip
                    )
                    self.opt_actor.step()

                    self.opt_critic.zero_grad()
                    loss_critic.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.critic.parameters(), self.grad_clip
                    )
                    self.opt_critic.step()

            if self.track:

                prefix = "Agent/"
                self.writer.add_scalar(prefix + "Loss/entropy", np.mean(entropy))
                self.writer.add_scalar(prefix + "Loss/actor", np.mean(actor_loss))
                self.writer.add_scalar(prefix + "Loss/critic", np.mean(critic_loss))
                self.writer.add_scalar(
                    prefix + "Action/std_mean", torch.mean(self.policy.log_action_std)
                )
                # self.params.writer.add_scalars(prefix+"Action/STD_Vals",{str(i):self.policy.log_action_var[i] for i in range(self.params.action_dim)},idx)
                self.writer.add_scalar(
                    prefix + "Loss/advantage_min", advantages.min().item()
                )
                self.writer.add_scalar(
                    prefix + "Loss/advantage_max", advantages.max().item()
                )
                # rewards_tensor = torch.stack(rewards)

                # rew_sum = torch.sum(rewards_tensor, dim=0)

                # mean_rew = float(torch.mean(rew_sum))

                self.writer.add_scalar(prefix + "Reward", torch.mean(rewards))

            # Copy new weights into old policy
            self.policy_old.load_state_dict(self.policy.state_dict())

            # Clear buffer
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


if __name__ == "__main__":
    pass
