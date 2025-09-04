from learning.environments.box2d_salp.domain import SalpChainEnv
import numpy as np
import pickle
import os


def biased_sample(env, zero_prob=1.0):
    """Sample from action space with biased link_openness"""
    action = env.action_space.sample()
    random_values = np.random.random(env.n_agents)
    action["link_openness"] = (random_values > zero_prob).astype(np.int8)
    return action


env = SalpChainEnv(render_mode="human", n_agents=8)

# Create a list to store info from each episode
info_record = []

for episode in range(25):
    obs, _ = env.reset()
    episode_info = None  # Will store the last info from this episode

    for step in range(1000):
        action = {
            "movement": np.random.uniform(-1, 1, size=(env.n_agents, 2)),
            "link_openness": np.random.randint(0, 2, size=env.n_agents),
            "detach": np.random.uniform(0, 1, size=env.n_agents),
        }

        obs, reward, terminated, truncated, info = env.step(action)
        episode_info = info  # Save the current info
        env.render()

        print(f"Step={step}, Reward={reward}, Terminated={terminated}")

        if terminated:
            print("TERMINATED")
            break

    # Save the last info from this episode
    if episode_info is not None:
        info_record.append(episode_info)

    print(f"Completed Episode {episode}")

env.close()

# Save the info_list using pickle
output_dir = os.path.dirname(os.path.abspath(__file__))
pickle_path = os.path.join(output_dir, "test_info.pkl")

with open(pickle_path, "wb") as f:
    pickle.dump(info_record, f)

print(f"Episode info saved to {pickle_path}")
