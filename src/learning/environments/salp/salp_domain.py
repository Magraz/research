#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import ValuesView
from vmas import render_interactively
from vmas.simulator.joints import Joint
from vmas.simulator.core import Agent, Landmark, Box, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils

from learning.environments.salp.world import SalpWorld
from learning.environments.salp.dynamics import SalpDynamics
from learning.environments.salp.controller import SalpController
from learning.environments.salp.sensors import SectorDensity
from learning.environments.salp.utils import (
    COLOR_LIST,
    generate_target_points,
    batch_discrete_frechet_distance,
    angle_between_vectors,
    is_within_any_range,
    closest_number,
)
from learning.environments.salp.types import Chain
import random
import math
from copy import deepcopy

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class SalpDomain(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # CONSTANTS
        self.agent_radius = 0.02
        self.agent_joint_length = 0.043
        self.agent_max_angle = 45
        self.agent_min_angle = -45
        self.u_multiplier = 1.0
        self.target_radius = self.agent_radius / 2

        # Environment
        self.x_semidim = kwargs.pop("x_semidim", 1)
        self.y_semidim = kwargs.pop("y_semidim", 1)
        self.viewer_zoom = kwargs.pop("viewer_zoom", 1.5)

        # Agents
        self.n_agents = kwargs.pop("n_agents", 2)
        self.starting_position = kwargs.pop("starting_position", [0.0, 0.0])
        self.state_representation = kwargs.pop("state_representation", "single")
        self.agents_colors = []
        self.agents_positions = []
        self.agents_idx = []
        self.agent_chains = []

        for idx in range(self.n_agents):
            self.agents_idx.append(idx)
            self.agents_colors.append(COLOR_LIST[idx])

        self.agent_starting_chains = [
            Chain(
                0,
                path=torch.tensor(
                    generate_target_points(
                        x=self.starting_position[0],
                        y=self.starting_position[1],
                        n_points=self.n_agents,
                        d_max=self.agent_joint_length,
                        theta_range=[0, 0],
                    ),
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .repeat(batch_dim, 1, 1),
                entities=[],
            )
        ]

        self.agent_chains = deepcopy(self.agent_starting_chains)

        # Targets
        self.n_targets = kwargs.pop("n_targets", 1)
        self.targets_start_positions = kwargs.pop(
            "targets_start_positions", [[0.0, 0.0]]
        )
        self.targets_colors = []
        self.target_chains = []

        for idx in range(self.n_targets):
            self.targets_colors.append(COLOR_LIST[idx])
            target_x = self.targets_start_positions[idx][0]
            target_y = self.targets_start_positions[idx][1]
            target_chain = Chain(
                idx,
                torch.tensor(
                    generate_target_points(
                        x=target_x,
                        y=target_y,
                        n_points=self.n_agents,
                        d_max=self.agent_joint_length,
                        theta_range=[self.agent_min_angle, self.agent_max_angle],
                    ),
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .repeat(batch_dim, 1, 1),
                entities=[],
            )
            self.target_chains.append(target_chain)

        if kwargs.pop("shuffle_agents_positions", False):
            random.shuffle(self.agents_idx)

        self._lidar_range = kwargs.pop("lidar_range", 10.0)

        # Reward Shaping
        self.frechet_shaping_factor = 1.0
        self.centroid_shaping_factor = 1.0

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.device = device

        self.covered_targets = torch.zeros((batch_dim, self.n_targets), device=device)

        # self.gravity_x_val = sample_filtered_normal(
        #     mean=0.0, std_dev=0.3, threshold=0.2
        # )

        gravity_x_vals = [0.4, -0.4]

        self.gravity_x_val = random.choice(gravity_x_vals)
        self.gravity_y_val = -0.3
        # Make world
        world = SalpWorld(
            batch_dim=batch_dim,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            device=device,
            substeps=15,
            collision_force=1500,
            joint_force=900,
            torque_constraint_force=0.01,
            # gravity=(
            #     self.gravity_x_val,
            #     self.gravity_y_val,
            # ),
        )

        # Set targets
        for target_chain in self.target_chains:
            for idx, _ in enumerate(target_chain.path[0, :, 0]):

                target = Landmark(
                    name=f"chain_{target_chain.idx}_target_{idx}",
                    shape=Sphere(radius=self.target_radius),
                    color=self.targets_colors[target_chain.idx],
                    collide=False,
                )

                target.chain_idx = target_chain.idx
                target.idx = idx

                world.add_landmark(target)
                target_chain.entities.append(target)

        # Add agents
        for agent_chain in self.agent_chains:
            for idx, _ in enumerate(agent_chain.path[0, :, 0]):

                agent = Agent(
                    name=f"chain_{agent_chain.idx}_agent_{idx}",
                    render_action=True,
                    shape=Box(
                        length=self.agent_radius * 2, width=self.agent_radius * 2.5
                    ),
                    dynamics=SalpDynamics(),
                    color=self.agents_colors[idx],
                    u_multiplier=self.u_multiplier,
                )

                agent.chain_idx = agent_chain.idx
                agent.idx = idx

                world.add_agent(agent)
                agent_chain.entities.append(agent)

        # Add joints
        self.joint_list = []
        for i in range(self.n_agents - 1):
            joint = Joint(
                world.agents[self.agents_idx[i]],
                world.agents[self.agents_idx[i + 1]],
                anchor_a=(0, 0),
                anchor_b=(0, 0),
                dist=self.agent_joint_length,
                rotate_a=False,
                rotate_b=False,
                collidable=False,
                width=0,
            )
            world.add_joint(joint)
            self.joint_list.append(joint)

        # Assign neighbors to agents
        for agent in world.agents:
            agent.state.local_neighbors = self.get_local_neighbors(agent, world.joints)
            agent.state.left_neighbors, agent.state.right_neighbors = (
                self.get_all_neighbors(agent, world.agents)
            )

        # Initialize reward tensors
        self.global_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.centroid_rew = self.global_rew.clone()
        self.frechet_rew = self.global_rew.clone()

        world.zero_grad()

        return world

    def reset_world_at(self, env_index: int = None):

        if env_index is None:
            self.all_time_covered_targets = torch.full(
                (self.world.batch_dim, self.n_targets),
                False,
                device=self.world.device,
            )
        else:
            self.all_time_covered_targets[env_index] = False

        for agent_chain in self.agent_chains:
            for idx, agent in enumerate(agent_chain.entities):

                pos = (
                    torch.ones(
                        (self.world.batch_dim, self.world.dim_p),
                        device=self.world.device,
                    )
                    * self.agent_starting_chains[0].path[:, idx, :]
                )

                agent.set_pos(
                    pos,
                    batch_index=env_index,
                )

                # agent.state.rot += -4 * torch.pi + torch.pi / 8

        for target_chain in self.target_chains:

            target_x = self.targets_start_positions[0][0] + random.uniform(-1, 1)
            target_y = self.targets_start_positions[0][1] + random.uniform(-1, 1)

            target_chain.path = (
                torch.tensor(
                    generate_target_points(
                        x=target_x,
                        y=target_y,
                        n_points=self.n_agents,
                        d_max=self.agent_joint_length,
                        theta_range=[self.agent_min_angle, self.agent_max_angle],
                    ),
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .repeat(self.world.batch_dim, 1, 1)
            )

            for idx, target in enumerate(target_chain.entities):

                pos = (
                    torch.ones(
                        (self.world.batch_dim, self.world.dim_p),
                        device=self.world.device,
                    )
                    * target_chain.path[:, idx, :]
                )

                target.set_pos(
                    pos,
                    batch_index=env_index,
                )

            target_chain.update()

        self.frechet_shaping = self.calculate_frechet_reward()
        self.centroid_shaping = self.calculate_centroid_reward()

    def interpolate(
        self,
        value,
        source_min=-1,
        source_max=1,
        target_min=-torch.pi,
        target_max=torch.pi,
    ):
        # Linear interpolation using PyTorch
        return target_min + (value - source_min) / (source_max - source_min) * (
            target_max - target_min
        )

    def process_action(self, agent: Agent):
        magnitude = (
            self.interpolate(agent.action.u[:, 0], target_min=0, target_max=1) * 0.2
        )
        turn_angle = torch.pi / 4
        in_theta = self.interpolate(
            agent.action.u[:, 1], target_min=-turn_angle, target_max=turn_angle
        )

        # Get heading vector
        agent_rot = agent.state.rot % (2 * torch.pi)
        heading_offset = agent_rot + torch.pi / 2

        theta = (in_theta + heading_offset) % (2 * torch.pi)

        theta_clamps = [
            (heading_offset - turn_angle) % (2 * torch.pi),
            (heading_offset + turn_angle) % (2 * torch.pi),
            # (heading_offset + torch.pi - segment_size) % (2 * torch.pi),
            # (heading_offset + torch.pi + segment_size) % (2 * torch.pi),
        ]
        theta_ranges = [
            (theta_clamps[0], theta_clamps[1]),
            # (theta_clamps[2], theta_clamps[3]),
        ]

        if is_within_any_range(theta, theta_ranges):
            x = torch.cos(theta).squeeze(-1) * magnitude
            y = torch.sin(theta).squeeze(-1) * magnitude
            agent.state.force = torch.stack((x, y), dim=-1)
        else:
            theta_clamp = closest_number(theta, theta_clamps)
            x = torch.cos(theta_clamp).squeeze(-1) * magnitude
            y = torch.sin(theta_clamp).squeeze(-1) * magnitude
            agent.state.force = torch.stack((x, y), dim=-1)

        # Join action
        # if agent.state.join.any():
        #     self.world.detach_joint(self.joint_list[0])

    def calculate_frechet_reward(self) -> torch.Tensor:

        f_dist = batch_discrete_frechet_distance(
            self.target_chains[0].path, self.agent_chains[0].path
        )
        f_rew = 1 / torch.exp(f_dist)

        return f_rew

    def calculate_centroid_reward(self) -> torch.Tensor:

        c_dist = torch.norm(
            self.target_chains[0].centroid - self.agent_chains[0].centroid,
            dim=1,
        )
        c_rew = 1 / torch.exp(c_dist)

        return c_rew

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        if is_first:

            # Calculate G
            self.agent_chains[0].update()

            self.frechet_rew[:] = 0
            self.centroid_rew[:] = 0

            f_rew = self.calculate_frechet_reward()
            frechet_shaping = f_rew * self.frechet_shaping_factor
            self.frechet_rew += frechet_shaping - self.frechet_shaping
            self.frechet_shaping = frechet_shaping

            cent_rew = self.calculate_centroid_reward()
            centroid_shaping = cent_rew * self.centroid_shaping_factor
            self.centroid_rew += centroid_shaping - self.centroid_shaping
            self.centroid_shaping = centroid_shaping

            self.total_rew = f_rew

            self.global_rew = self.frechet_rew

        if is_last:
            self.current_order_per_env = torch.sum(
                self.all_time_covered_targets, dim=1
            ).unsqueeze(-1)

        rew = torch.cat([self.global_rew * 10])

        return rew

    def get_local_neighbors(self, agent: Agent, joints: ValuesView):
        neighbors = []
        links = []

        # Get links
        for joint in joints:

            if agent == joint.entity_a:
                links.append(joint.entity_b)
            elif agent == joint.entity_b:
                links.append(joint.entity_a)

        # Get agents
        for joint in joints:

            if (joint.entity_a in links) and (joint.entity_b != agent):
                neighbors.append(joint.entity_b)
            elif (joint.entity_b in links) and (joint.entity_a != agent):
                neighbors.append(joint.entity_a)

        return neighbors

    def get_all_neighbors(self, agent, all_agents):
        l_neighbors = []
        r_neighbors = []

        for a in all_agents:
            if a != agent:
                if agent.idx < a.idx:
                    r_neighbors.append(a)
                else:
                    l_neighbors.append(a)

        return l_neighbors, r_neighbors

    def get_heading(self, agent: Agent):
        x = torch.cos(agent.state.rot + 1.5 * torch.pi).squeeze(-1)
        y = torch.sin(agent.state.rot + 1.5 * torch.pi).squeeze(-1)

        return torch.stack((x, y), dim=-1)

    def calculate_moment(self, position, force):
        """
        Calculate the moment generated by a 2D force at a given position using PyTorch.

        Parameters:
            position (torch.Tensor): Tensor of shape [N, 2], where each row is (x_r, y_r).
            force (torch.Tensor): Tensor of shape [N, 2], where each row is (x_f, y_f).

        Returns:
            torch.Tensor: Tensor of shape [N] containing the moment for each pair of position and force.
        """
        # Ensure tensors are of the same shape
        assert (
            position.shape == force.shape
        ), "Position and force tensors must have the same shape."

        # Extract components
        x_r, y_r = position[:, 0], position[:, 1]
        x_f, y_f = force[:, 0], force[:, 1]

        # Compute the moment
        moment = x_r * y_f - y_r * x_f

        return moment

    def single_agent_representation(self, agent: Agent):

        self.agent_chains[0].update()

        agent_vect_2_agent_centroid = agent.state.pos - self.agent_chains[0].centroid

        agent_vect_2_target_centroid = agent.state.pos - self.target_chains[0].centroid

        f_dist = batch_discrete_frechet_distance(
            self.agent_chains[0].path, self.target_chains[0].path
        )

        c_dist = torch.norm(
            self.target_chains[0].centroid - self.agent_chains[0].centroid,
            dim=1,
        )

        total_moment = 0
        total_force = 0
        for a in self.world.agents:
            if a != agent:
                r = self.agent_chains[0].centroid - agent.state.pos
                total_moment += self.calculate_moment(r, agent.state.force)
                total_force += agent.state.force

        observation = torch.cat(
            [
                total_moment.unsqueeze(-1),
                # total_force,
                f_dist.unsqueeze(-1),
                agent_vect_2_agent_centroid,
                # agent_vect_2_target_centroid,
                agent.state.pos,
                agent.state.vel,
                agent.state.rot % (2 * torch.pi),
                agent.state.ang_vel,
            ],
            dim=-1,
        )

        return observation

    def observation(self, agent: Agent):
        return self.single_agent_representation(agent)

    def done(self):
        return self.total_rew > 0.98

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "global_reward": (self.global_rew),
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []
        # Target ranges
        for target_chain in self.target_chains:
            for target in target_chain.entities:
                range_circle = rendering.make_circle(self.target_radius, filled=False)
                xform = rendering.Transform()
                xform.set_translation(*target.state.pos[env_index])
                range_circle.add_attr(xform)
                range_circle.set_color(*self.targets_colors[target_chain.idx].value)
                geoms.append(range_circle)

        return geoms

    def random_point_around_center(center_x, center_y, radius):
        """
        Generates a random (x, y) coordinate around a given circle center within the specified radius.

        Parameters:
            center_x (float): The x-coordinate of the circle center.
            center_y (float): The y-coordinate of the circle center.
            radius (float): The radius around the center where the point will be generated.

        Returns:
            tuple: A tuple (x, y) representing the random point.
        """
        # Generate a random angle in radians
        angle = random.uniform(0, 2 * math.pi)
        # Generate a random distance from the center, within the circle
        distance = random.uniform(0, radius)

        # Calculate the x and y coordinates
        random_x = center_x + distance * math.cos(angle)
        random_y = center_y + distance * math.sin(angle)

        return [random_x, random_y]


if __name__ == "__main__":
    render_interactively(__file__, joints=True, control_two_agents=True)
