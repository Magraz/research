from learning.algorithms.ppo.ppo import PPO
from learning.algorithms.ppo.types import Params


class IPPO:
    def __init__(self, params: Params):
        self.params = params
        self.n_agents = params.n_agents
        self.shared_params = params.shared_params

        if not self.shared_params:
            self.agents = [PPO(params, n_buffers=1) for i in range(self.n_agents)]
        else:
            self.shared_policy = PPO(params, n_buffers=self.n_agents)
            self.agents = [self.shared_policy for _ in range(self.n_agents)]

    def act(self, states):
        action = []
        for i, (agent, state) in enumerate(zip(self.agents, states)):
            action.append(agent.select_action(state, n_buffer=i))
        return action

    def act_deterministic(self, states):
        action = []
        for agent, state in zip(self.agents, states):
            action.append(agent.deterministic_action(state))
        return action

    def add_reward_terminal(self, rewards, term):
        for i, (agent, reward) in enumerate(zip(self.agents, rewards)):
            agent.add_reward_terminal(reward, term, n_buffer=i)

    def train(self):
        if not self.shared_params:
            for agent in self.agents:
                agent.update()
        else:
            self.shared_policy.update()

    def save(self, fname):
        for i, agent in zip(range(self.n_agents), self.agents):
            agent.save(f"{str(fname)}_a{str(i)}")

    def load(self, fname):
        for i, agent in zip(range(self.n_agents), self.agents):
            agent.load(f"{str(fname)}_a{str(i)}")
