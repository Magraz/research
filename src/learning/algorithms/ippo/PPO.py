import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np

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



class ActorCritic(nn.Module):
	def __init__(self, params):
		super(ActorCritic, self).__init__()
		state_dim = params.state_dim 
		action_dim = params.action_dim
		action_std_init = params.action_std
		actor_hidden = params.actor_hidden
		critic_hidden = params.critic_hidden
		active_fn = params.active_fn
		self.device=params.device
		
		self.var_learned=params.var_learned
		self.action_dim = action_dim
		self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
		self.log_action_var =nn.Parameter(torch.rand(action_dim)/100+params.action_std).to(self.device)
		# actor
		self.actor = nn.Sequential(
						nn.Linear(state_dim, actor_hidden),
						active_fn(),
						nn.Linear(actor_hidden, actor_hidden),
						active_fn(),
						nn.Linear(actor_hidden, action_dim),
						nn.Tanh()
					)
		
		
		# critic
		self.critic = nn.Sequential(
						nn.Linear(state_dim, critic_hidden),
						active_fn(),
						nn.Linear(critic_hidden, critic_hidden),
						active_fn(),
						nn.Linear(critic_hidden, 1)
					)
		
	def set_action_std(self, new_action_std):
		self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

	def forward(self):
		raise NotImplementedError
	
	def act(self, state,deterministic=False):


		action_mean = self.actor(state)
		if self.var_learned:
			action_var=torch.exp(self.log_action_var)
		else:
			action_var=self.action_var
		dist = Normal(action_mean,action_var)
		action = dist.sample()
		action_logprob = dist.log_prob(action)
		state_val = self.critic(state)

		if deterministic:
			return action_mean.detach()
		else:
			return action.detach(), action_logprob.detach(), state_val.detach()
	
	def evaluate(self, state, action):

		action_mean = self.actor(state)
		if self.var_learned:
			action_var=torch.exp(self.log_action_var)
		else:
			action_var=self.action_var
		dist = Normal(action_mean,action_var)
		action_logprobs = dist.log_prob(action)
		dist_entropy = dist.entropy()
		state_values = self.critic(state)
		
		return action_logprobs, state_values, dist_entropy


class PPO:
	def __init__(self, params,idx=0):

		self.params=params
		self.device=params.device
		#self.action_std = params.action_std

		self.gamma = params.gamma
		self.eps_clip = params.eps_clip
		self.K_epochs = params.K_epochs
		
		self.buffer = RolloutBuffer()

		self.policy = ActorCritic(params).to(self.device)
		self.opt_actor = torch.optim.Adam([p for p in self.policy.actor.parameters()]+[self.policy.log_action_var],lr=params.lr_actor)
		self.opt_critic = torch.optim.Adam(self.policy.critic.parameters(),lr=params.lr_critic)

		self.policy_old = ActorCritic(params).to(self.device)
		self.policy_old.load_state_dict(self.policy.state_dict())
		
		self.MseLoss = nn.MSELoss()
		self.decay_action_std(0)
		self.idx=idx


	def set_action_std(self, new_action_std):
		self.action_std = new_action_std
		self.policy.set_action_std(new_action_std)
		self.policy_old.set_action_std(new_action_std)
	
	

	def decay_action_std(self,idx):
		percent=float(idx)/1.0e5
		#val=max(self.params.action_std-(self.params.decay_rate*percent),0.1)
		val=np.exp(self.params.action_std-(self.params.decay_rate*percent))
		self.set_action_std(val)
		
	def select_action(self, state):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device)
			action, action_logprob, state_val = self.policy_old.act(state)

		self.buffer.states.append(state)
		self.buffer.actions.append(action)
		self.buffer.logprobs.append(action_logprob)
		self.buffer.state_values.append(state_val)

		return action.detach().cpu().numpy().flatten()

	def deterministic_action(self, state):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device)
			action = self.policy_old.act(state,deterministic=True)

		return action.detach().cpu().numpy().flatten()
	
	def add_reward_terminal(self,reward,done):
		self.buffer.rewards.append(float(reward))
		self.buffer.is_terminals.append(done)

 
	def gae(self,rewards,values, is_terminals ):
		returns = []
		gae = 0
		for i in reversed(range(len(rewards))):
			if is_terminals[i]:
				delta = rewards[i]  - values[i]
				gae = delta 
			else:
				delta = rewards[i] + self.params.gamma * values[i + 1]  - values[i]
				gae = delta + self.params.gamma * self.params.lmbda * gae
			returns.append([(gae).item()])
		returns = [r for r in reversed(returns)]
		adv = np.array(returns)
		adv=(adv - np.mean(adv)) / (np.std(adv) + 1e-10)
		return torch.from_numpy(adv).to(self.device)

	def update(self,idx):
		# Monte Carlo estimate of returns
		self.decay_action_std(idx)
		rewards = []
		discounted_reward = 0
		for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.gamma * discounted_reward)
			rewards.insert(0, discounted_reward)
			
		# Normalizing the rewards
		rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

		#rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		# convert list to tensor
		old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
		old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
		old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
		old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

		# calculate advantages
		#advantages = (rewards.detach() - old_state_values.detach()).reshape((-1,1))
		#advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

		advantages=self.gae(self.buffer.rewards,self.buffer.state_values,self.buffer.is_terminals)
		
		Aloss,Closs,Entropy=[],[],[]
		# Optimize policy for K epochs
		for _ in range(self.K_epochs):

			# Evaluating old actions and values
			logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

			# match state_values tensor dimensions with rewards tensor
			state_values = torch.squeeze(state_values)
			
			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs.detach())

			# Finding Surrogate Loss  
			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

			# final loss of clipped objective PPO
			loss_actor = -torch.min(surr1, surr2)
			if self.params.var_learned:
				loss_actor=loss_actor - self.params.beta_ent * dist_entropy
			loss_actor=loss_actor.mean()
			loss_critic= self.MseLoss(state_values, rewards)
			Entropy.append(dist_entropy.mean().item())
			Aloss.append(loss_actor.item())
			Closs.append(loss_critic.item())
			# take gradient step
			self.opt_actor.zero_grad()
			loss_actor.backward()
			torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.params.grad_clip)
			self.opt_actor.step()

			self.opt_critic.zero_grad()
			loss_critic.backward()
			torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.params.grad_clip)
			self.opt_critic.step()
		if self.params.log_indiv:
			prefix="Agent"+str(self.idx)+"/"
			self.params.writer.add_scalar(prefix+"Loss/entropy", np.mean(Entropy),idx)
			self.params.writer.add_scalar(prefix+"Loss/actor", np.mean(Aloss),idx)
			self.params.writer.add_scalar(prefix+"Loss/critic", np.mean(Closs),idx)	
			self.params.writer.add_scalar(prefix+"Acton_std", self.action_std,idx)	
			self.params.writer.add_scalar(prefix+"Action/STD_Mean",torch.mean(self.policy.log_action_var),idx)	
			#elf.params.writer.add_scalars(prefix+"Action/STD_Vals",{str(i):self.policy.log_action_var[i] for i in range(self.params.action_dim)},idx)	
			self.params.writer.add_scalar(prefix+"Loss/Advantage_min", min(advantages),idx)
			self.params.writer.add_scalar(prefix+"Loss/Advantage_max", max(advantages),idx)
			self.params.writer.add_scalar(prefix+"Reward", sum(self.buffer.rewards)/self.params.N_batch,idx)

		# Copy new weights into old policy
		self.policy_old.load_state_dict(self.policy.state_dict())

		# clear buffer
		self.buffer.clear()
	
	def save(self, checkpoint_path):
		torch.save(self.policy_old.state_dict(), checkpoint_path)
   
	def load(self, checkpoint_path):
		self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
		self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

class Params:
	def __init__(self,fname=None,n_agents=0):	
		self.K_epochs = 20			   # update policy for K epochs in one PPO update
		self.N_batch=8
		self.N_steps=3e6
		self.eps_clip = 0.2		  # clip parameter for PPO
		self.gamma = 0.99			# discount factor

		self.lr_actor = 0.0003	   # learning rate for actor network
		self.lr_critic = 0.001	   # learning rate for critic network
		self.action_std = -0.8	  
		self.decay_rate=0.07      #per 100k steps
		self.random_seed = 0
		self.grad_clip=1.0

		self.action_dim = 4
		self.state_dim = 24

		self.actor_hidden = 64
		self.critic_hidden = 64
		self.active_fn = nn.LeakyReLU
		#self.active_fn = nn.Tanh
		
		self.lmbda=0.95

		self.max_steps=1000
		self.device="cpu"
		self.log_indiv=True
		if fname is not None:
			self.writer=SummaryWriter("./logs/"+fname)
		else:
			self.log_indiv=False
		self.var_learned=True
		self.beta_ent=0.001
		self.n_agents=n_agents

	def write(self):
		for key,val in self.__dict__.items():
			self.writer.add_text("Params/"+key, key+" : "+str(val))

if __name__ =="__main__":
	pass