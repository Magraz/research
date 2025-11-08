from learning.environments.types import EnvironmentEnum, EnvironmentParams


def create_env(env_config: EnvironmentParams, render_mode: str = None):

    match (env_config.environment):
        case EnvironmentEnum.BOX2D_SALP:
            from learning.environments.box2d_salp.domain import SalpChainEnv

            # Environment configuration
            env = SalpChainEnv(n_agents=env_config.n_agents, render_mode=render_mode)
            state_dim = env.observation_space.shape[1]
            action_dim = 4

        case EnvironmentEnum.MPE_SPREAD:
            from mpe2 import simple_spread_v3

            env = simple_spread_v3.parallel_env(
                N=env_config.n_agents,
                local_ratio=0.5,
                max_cycles=25,
                continuous_actions=False,
                dynamic_rescaling=True,
                render_mode=render_mode,
            )
            state_dim = env.observation_space("agent_0").shape[0]
            action_dim = env.action_space("agent_0").n

        case EnvironmentEnum.MPE_SIMPLE:
            from mpe2 import simple_v3

            env = simple_v3.parallel_env(
                max_cycles=25,
                continuous_actions=False,
                render_mode=render_mode,
            )
            state_dim = env.observation_space("agent_0").shape[0]
            action_dim = env.action_space("agent_0").n

    return env, state_dim, action_dim
