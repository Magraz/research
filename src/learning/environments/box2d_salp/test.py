from learning.environments.box2d_salp.domain import SalpChainEnv

import numpy as np

env = SalpChainEnv(render_mode="human", n_agents=8)
obs, _ = env.reset()

for _ in range(2000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

env.close()
