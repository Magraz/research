from PPO import PPO
import numpy as np

class IPPO:
    def __init__(self,params):
        self.params=params
        self.n_agents=params.n_agents
        self.agents=[PPO(params,i) for i in range(self.n_agents)]
    
    def act(self,states):
        action=[]
        for agent,state in zip(self.agents,states):
            action.append(agent.select_action(state))
        return np.array(action)
    
    def act_deterministic(self,states):
        action=[]
        for agent,state in zip(self.agents,states):
            action.append(agent.deterministic_action(state))
        return np.array(action)
    def add_reward_terminal(self,rewards,term):
        for agent,reward in zip(self.agents,rewards):
            agent.add_reward_terminal(reward,term)
        
    def train(self,idx):
        for agent in self.agents:
            agent.update(idx)

    def save(self,fname):
        for i,agent in zip(range(self.n_agents),self.agents):
            agent.save(fname+str(i))
            
    def load(self,fname):
        for i,agent in zip(range(self.n_agents),self.agents):
            agent.load(fname+str(i))
            
