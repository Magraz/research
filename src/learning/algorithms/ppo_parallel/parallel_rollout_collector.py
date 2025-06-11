import torch.multiprocessing as mp
from learning.algorithms.ppo_parallel.env_runner import EnvRunner

from multiprocessing import Process
import dill


class DillProcess(Process):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target = dill.dumps(
            self._target
        )  # Save the target function as bytes, using dill

    def run(self):
        if self._target:
            self._target = dill.loads(
                self._target
            )  # Unpickle the target function before executing
            self._target(*self._args, **self._kwargs)  # Execute the target function


class ParallelRolloutCollector:
    def __init__(self, policy, env_creator, n_envs_per_worker, n_workers, device):
        self.policy = policy
        self.n_workers = n_workers
        self.device = device

        # Create environments for each worker
        self.env_runners = []
        envs_per_worker = n_envs_per_worker
        for i in range(n_workers):
            # Each worker gets its own VMAS environment batch
            env = env_creator(envs_per_worker, base_seed=i * envs_per_worker)
            self.env_runners.append(EnvRunner(policy, env, device))

        # Use spawn method which is more compatible with complex objects
        mp_ctx = mp.get_context("spawn")
        self.pool = mp_ctx.Pool(processes=n_workers)

    def collect_rollouts(self, steps_per_env):
        # Run collection across multiple workers without passing self.env_runners
        # Instead use direct worker_id-based collection
        results = []

        # For parallel collection
        worker_args = [(i, steps_per_env) for i in range(self.n_workers)]
        results = self.pool.starmap(self._run_worker, worker_args)

        # Combine results from all workers
        combined_buffer = self._combine_buffers(results)
        return combined_buffer

    def _run_worker(self, worker_id, steps):
        """Run a specific worker directly"""
        return self.env_runners[worker_id].run(steps)

    def _combine_buffers(self, buffers):
        """Combines TensorRolloutBuffers from multiple workers into one."""
        from learning.algorithms.ppo_parallel.env_runner import TensorRolloutBuffer

        # Get dimensions from first buffer
        steps = buffers[0].states.shape[0]
        total_envs = sum(buffer.states.shape[1] for buffer in buffers)
        n_agents = buffers[0].states.shape[2]
        d_state = buffers[0].states.shape[3]
        d_action = buffers[0].actions.shape[3]

        # Create a combined buffer
        combined = TensorRolloutBuffer(
            steps, total_envs, n_agents, d_state, d_action, self.device
        )

        # Track the starting environment index for each worker's contribution
        env_offset = 0

        # For each buffer from each worker
        for buffer in buffers:
            worker_envs = buffer.states.shape[1]

            # Copy data for this worker's environments
            combined.states[:, env_offset : env_offset + worker_envs] = buffer.states
            combined.actions[:, env_offset : env_offset + worker_envs] = buffer.actions
            combined.logprobs[:, env_offset : env_offset + worker_envs] = (
                buffer.logprobs
            )
            combined.rewards[:, env_offset : env_offset + worker_envs] = buffer.rewards
            combined.values[:, env_offset : env_offset + worker_envs] = buffer.values
            combined.is_terminals[:, env_offset : env_offset + worker_envs] = (
                buffer.is_terminals
            )

            # Update the offset for the next worker's data
            env_offset += worker_envs

        combined.step = steps  # Mark the buffer as fully filled
        return combined
