from learning.environments.box2d_salp.domain import SalpChainEnv
import numpy as np

env = SalpChainEnv(render_mode="human", n_agents=10)
obs, _ = env.reset()

for step in range(5000):
    # if step < 100:
    #     # Push agents in different directions with 2D forces
    #     action = np.array(
    #         [
    #             [1.0, 0.5],  # Agent 0: right and up
    #             [-0.5, 1.0],  # Agent 1: left and up
    #             [0.0, -1.0],  # Agent 2: down only
    #             [1.0, -0.5],  # Agent 3: right and down
    #             [-1.0, 0.0],  # Agent 4: left only
    #         ]
    #     )

    # if step < 200:
    #     # Circular motion pattern
    #     t = (step - 100) * 0.1
    #     action = np.array(
    #         [
    #             [np.cos(t + i * np.pi / 4), np.sin(t + i * np.pi / 4)]
    #             for i in range(env.n_agents)
    #         ]
    #     )

    # Random 2D actions
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    print(f"Step {step}: Reward = {reward}")

env.close()
