import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from learning.algorithms.ppo.types import Params

import dill

# Useful for error tracing
torch.autograd.set_detect_anomaly(True)


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        del self.is_terminals[:]


class RolloutData(Dataset):
    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        values: torch.Tensor,
        logprobs: torch.Tensor,
        advantages: torch.Tensor,
        rewards: torch.Tensor,
    ):
        self.states = states
        self.actions = actions
        self.values = values
        self.logprobs = logprobs
        self.advantages = advantages
        self.rewards = rewards

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.values[idx],
            self.logprobs[idx],
            self.advantages[idx],
            self.rewards[idx],
        )


class PPO:
    def __init__(
        self,
        device: str,
        model: str,
        params: Params,
        n_agents_train: int,
        n_agents_eval: int,
        n_envs: int,
        d_state: int,
        d_action: int,
        writer: SummaryWriter = None,
        checkpoint: bool = False,
    ):
        self.device = device
        self.writer = writer
        self.checkpoint = checkpoint
        self.n_envs = n_envs
        self.n_agents = n_agents_train
        self.d_action = d_action
        self.buffer = RolloutBuffer()

        # Algorithm parameters
        self.n_epochs = params.n_epochs
        self.minibatch_size = params.minibatch_size
        self.gamma = params.gamma
        self.lmbda = params.lmbda
        self.eps_clip = params.eps_clip
        self.grad_clip = params.grad_clip
        self.ent_coef = params.ent_coef
        self.std_coef = params.std_coef
        self.n_epochs = params.n_epochs
        self.use_clipped_value_loss = True

        # Select model
        match (model):
            case "mlp":
                from learning.algorithms.ppo.models.mlp_ac import ActorCritic
            case "transformer":
                from learning.algorithms.ppo.models.transformer_ac import ActorCritic
            case "gat":
                from learning.algorithms.ppo.models.gat_ac import ActorCritic

        # Create models
        self.policy = ActorCritic(
            n_agents_train,
            n_agents_eval,
            d_state,
            d_action,
            self.device,
        ).to(self.device)

        self.policy_old = ActorCritic(
            n_agents_train,
            n_agents_eval,
            d_state,
            d_action,
            self.device,
        ).to(self.device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        # Create optimizers
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=params.lr_actor,
        )

        # Logging params
        self.total_epochs = 0

        if self.checkpoint:
            # Load epoch count
            path = self.writer.log_dir / "tensorboard.dat"
            if path.is_file():
                with open(path, "rb") as file:
                    self.total_epochs = dill.load(file)["total_epochs"]

    def calc_value_loss(self, values, value_preds_batch, return_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """

        clip_param = 0.05
        huber_delta = 10

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -clip_param, clip_param
        )

        value_loss_clipped = F.huber_loss(
            return_batch, value_pred_clipped, delta=huber_delta
        )
        value_loss_original = F.huber_loss(return_batch, values, delta=huber_delta)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss

    def select_action(self, state: torch.Tensor):
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.values.append(state_val)

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
        gae = torch.zeros(self.n_envs, device=self.device)

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
        final_values = self.policy_old.get_value(
            self.buffer.states[-1]
        ) * ~self.buffer.is_terminals[-1].unsqueeze(-1)

        bootstrapped_values = self.buffer.values.copy()
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
        old_values = (
            torch.stack(self.buffer.values).transpose(1, 0).flatten(end_dim=1).detach()
        )
        old_logprobs = (
            torch.stack(self.buffer.logprobs)
            .transpose(1, 0)
            .flatten(end_dim=1)
            .detach()
        )

        # Create dataset from rollout
        dataset = RolloutData(
            old_states, old_actions, old_values, old_logprobs, advantages, returns
        )
        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        # Load model into GPU for training
        train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.policy.to(train_device)
            self.policy.device = train_device

        # Optimize policy for n epochs
        for _ in range(self.n_epochs):

            for (
                b_old_states,
                b_old_actions,
                b_old_values,
                b_old_logprobs,
                b_advantages,
                b_returns,
            ) in loader:

                # Load batch into GPU for training
                if torch.cuda.is_available():
                    b_old_states = b_old_states.to(train_device)
                    b_old_actions = b_old_actions.to(train_device)
                    b_old_values = b_old_values.to(train_device)
                    b_old_logprobs = b_old_logprobs.to(train_device)
                    b_advantages = b_advantages.to(train_device)
                    b_returns = b_returns.to(train_device)

                # Evaluating old actions and values
                logprobs, values, dist_entropy = self.policy.evaluate(
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
                    self.std_coef
                    * self.policy.log_action_std[: self.n_agents * self.d_action]
                    .square()
                    .mean()
                )

                # Promote exploration by reducing the loss if entropy increases
                entropy_bonus = self.ent_coef * dist_entropy.mean()

                ppo_loss = -torch.min(surr1, surr2).mean()

                # Calculate actor and critic losses
                actor_loss = ppo_loss + log_std_penalty - entropy_bonus
                value_loss = self.calc_value_loss(values, b_old_values, b_returns)

                loss = actor_loss + value_loss

                # Take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                self.optimizer.step()

                # Store data
                if self.writer is not None:
                    self.writer.add_scalar(
                        "Agent/actor_loss", actor_loss.item(), self.total_epochs
                    )
                    self.writer.add_scalar(
                        "Agent/critic_loss", value_loss.item(), self.total_epochs
                    )
                    self.writer.add_scalar(
                        "Agent/entropy", dist_entropy.mean().item(), self.total_epochs
                    )
                    self.writer.add_scalar(
                        "Agent/log_prob", logprobs.mean().item(), self.total_epochs
                    )
                    self.writer.add_scalar(
                        "Agent/log_action_std",
                        self.policy.log_action_std[: self.n_agents * self.d_action]
                        .mean()
                        .item(),
                        self.total_epochs,
                    )

                    self.total_epochs += 1

                    # Store epoch count
                    with open(self.writer.log_dir / "tensorboard.dat", "wb") as f:
                        log_data_dict = {"total_epochs": self.total_epochs}
                        dill.dump(log_data_dict, f)

        # Load model back to cpu to collect rollouts
        self.policy.to(self.device)
        self.policy.device = self.device

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )
        )
        self.policy.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )
        )
