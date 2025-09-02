from learning.environments.box2d_salp.domain import SalpChainEnv
import numpy as np


def biased_sample(env, zero_prob=1.0):
    """Sample from action space with biased link_openness"""
    action = env.action_space.sample()
    random_values = np.random.random(env.n_agents)
    action["link_openness"] = (random_values > zero_prob).astype(np.int8)
    return action


env = SalpChainEnv(render_mode="human", n_agents=12)

for episodes in range(10):
    obs, _ = env.reset()
    for step in range(100):

        action = {
            "movement": np.random.uniform(-1, 1, size=(env.n_agents, 2)),
            "link_openness": np.random.randint(0, 2, size=env.n_agents),
            "detach": np.random.uniform(0, 1, size=env.n_agents),
        }

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        print(f"Step={step}, Reward={reward}, Terminated={terminated}")

        # if terminated == True:
        #     print("TERMINATED")
        #     break

env.close()
