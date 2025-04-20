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
from vmas.simulator.core import Agent, Landmark, Box, Sphere, World, Line
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils

from learning.environments.salp.world import SalpWorld
from learning.environments.salp.dynamics import SalpDynamics
from learning.environments.salp.utils import (
    COLOR_LIST,
    COLOR_MAP,
    generate_target_points,
    batch_discrete_frechet_distance,
    angular_velocity,
    generate_random_coordinate_outside_box,
    rotate_points,
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
        self.agent_joint_length = 0.052
        self.agent_max_angle = 45
        self.agent_min_angle = -45
        self.u_multiplier = 0.6
        self.target_radius = self.agent_radius / 2

        # Environment
        self.x_semidim = kwargs.pop("x_semidim", 1)
        self.y_semidim = kwargs.pop("y_semidim", 1)
        self.viewer_zoom = kwargs.pop("viewer_zoom", 1.05)

        # Agents
        self.n_agents = kwargs.pop("n_agents", 2)
        self.state_representation = kwargs.pop("state_representation", "global")
        self.agent_chains = [None for _ in range(batch_dim)]

        # Targets
        self.n_targets = kwargs.pop("n_targets", 1)
        self.target_chains = [None for _ in range(batch_dim)]

        if kwargs.pop("shuffle_agents_positions", False):
            random.shuffle(self.agents_idx)

        self._lidar_range = kwargs.pop("lidar_range", 10.0)

        # Reward Shaping
        self.frechet_shaping_factor = 1.0
        self.centroid_shaping_factor = 1.0

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.device = device
        # Make world
        world = SalpWorld(
            batch_dim=batch_dim,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            device=device,
            substeps=15,
            collision_force=1500,
            joint_force=900,
            contact_margin=3e-3,
        )

        # Set targets
        self.targets = []
        for n_target in range(self.n_targets):
            for n_agent in range(self.n_agents):
                target = Landmark(
                    name=f"target_{n_agent}_chain_{n_target}",
                    shape=Sphere(radius=self.target_radius),
                    color=COLOR_MAP["RED"],
                    collide=False,
                )
                world.add_landmark(target)
                self.targets.append(target)

        # Add agents
        self.agents = []
        for n_agent in range(self.n_agents):
            agent = Agent(
                name=f"agent_{n_agent}",
                render_action=True,
                shape=Box(length=self.agent_radius * 2, width=self.agent_radius * 2.5),
                dynamics=SalpDynamics(),
                color=random.choice(COLOR_LIST),
                u_multiplier=self.u_multiplier,
            )
            world.add_agent(agent)
            self.agents.append(agent)

        # Add joints
        self.joints = []
        for i in range(self.n_agents - 1):
            joint = Joint(
                world.agents[i],
                world.agents[i + 1],
                anchor_a=(0, 0),
                anchor_b=(0, 0),
                dist=self.agent_joint_length,
                rotate_a=True,
                rotate_b=True,
                collidable=False,
                width=0,
            )
            world.add_joint(joint)
            self.joints.append(joint)

        # Assign neighbors to agents
        # for agent in world.agents:
        #     agent.state.local_neighbors = self.get_local_neighbors(agent, world.joints)
        #     agent.state.left_neighbors, agent.state.right_neighbors = (
        #         self.get_all_neighbors(agent, world.agents)
        #     )

        # Initialize reward tensors
        self.global_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.centroid_rew = self.global_rew.clone()
        self.frechet_rew = self.global_rew.clone()

        world.zero_grad()

        return world

    def reset_world_at(self, env_index: int = None):
        joint_delta_x = self.agent_joint_length / 2
        joint_delta_y = 0.0
        agent_scale = 0.1
        agent_offset = 0.0

        target_offset = self.x_semidim - abs(self.agent_joint_length * self.n_agents)
        target_scale = self.x_semidim / 100

        if env_index is None:
            # Create new agent and target chains
            self.agent_chains = [
                self.create_new_chain(
                    offset=agent_offset, scale=agent_scale, theta_min=0.0, theta_max=0.0
                )
                for _ in range(self.world.batch_dim)
            ]

            self.target_chains = [
                self.create_new_chain(
                    offset=target_offset,
                    scale=target_scale,
                    theta_min=self.agent_min_angle,
                    theta_max=self.agent_max_angle,
                    rotation_angle=math.radians(random.uniform(0, 359)),
                )
                for _ in range(self.world.batch_dim)
            ]

            # Set positions according to chains
            agent_chain_tensor = torch.stack(
                [agent_chain.path for agent_chain in self.agent_chains]
            )
            for i, agent in enumerate(self.agents):
                pos = agent_chain_tensor[:, i, :]
                agent.set_pos(pos, batch_index=env_index)

            target_chain_tensor = torch.stack(
                [target_chain.path for target_chain in self.target_chains]
            )
            for i, target in enumerate(self.targets):
                pos = target_chain_tensor[:, i, :]
                target.set_pos(pos, batch_index=env_index)

            joint_delta = torch.tensor(
                (joint_delta_x, joint_delta_y), device=self.device
            ).repeat(self.world.batch_dim, 1)

            for i, joint in enumerate(self.joints):
                joint.landmark.set_pos(
                    self.agents[i].state.pos + joint_delta,
                    batch_index=env_index,
                )

        else:
            self.agent_chains[env_index] = self.create_new_chain(
                offset=agent_offset, scale=agent_scale, theta_min=0.0, theta_max=0.0
            )
            self.target_chains[env_index] = self.create_new_chain(
                offset=target_offset,
                scale=target_scale,
                theta_min=self.agent_min_angle,
                theta_max=self.agent_max_angle,
                rotation_angle=math.radians(random.uniform(0, 359)),
            )

            for n_agent, agent in enumerate(self.world.agents):
                pos = self.agent_chains[env_index].path[n_agent]
                agent.set_pos(pos, batch_index=env_index)

            for n_target, target in enumerate(self.targets):
                pos = self.target_chains[env_index].path[n_target]
                target.set_pos(pos, batch_index=env_index)

            joint_delta = torch.tensor(
                (joint_delta_x, joint_delta_y), device=self.device
            )

            for i, joint in enumerate(self.joints):
                joint.landmark.set_pos(
                    self.agents[i].state.pos[env_index] + joint_delta,
                    batch_index=env_index,
                )

        self.frechet_shaping = self.calculate_frechet_reward()
        self.centroid_shaping = self.calculate_centroid_reward()

    def is_out_of_bounds(self):
        """Boolean mask of shape (n_envs,) – True if agent is out of bounds."""
        out_of_bounds = []

        for agent in self.agents:
            pos = agent.state.pos  # (n_envs, 2)
            x_ok = pos[..., 0].abs() <= self.world.x_semidim - 1e-4
            y_ok = pos[..., 1].abs() <= self.world.y_semidim - 1e-4
            out_of_bounds.append(~(x_ok & y_ok))

        out_of_bounds = torch.stack(out_of_bounds).transpose(1, 0).any(dim=-1)

        return out_of_bounds

    def create_new_chain(
        self, offset, scale, theta_min, theta_max, rotation_angle: float = 0.0
    ):
        x_coord, y_coord = generate_random_coordinate_outside_box(
            offset,
            scale,
            self.x_semidim,
            self.y_semidim,
        )
        chain = Chain(
            path=rotate_points(
                points=generate_target_points(
                    x=x_coord,
                    y=y_coord,
                    n_points=self.n_agents,
                    d_max=self.agent_joint_length,
                    theta_range=[
                        theta_min,
                        theta_max,
                    ],
                ),
                angle_rad=rotation_angle,
            ),
        )
        return chain

    def create_chain_from_agents(self, n_env):
        agents_pos = [agent.state.pos[n_env] for agent in self.world.agents]
        chain = Chain(path=torch.stack(agents_pos))
        return chain

    def update_agent_chains(self):
        self.agent_chains = [
            self.create_chain_from_agents(n_env)
            for n_env in range(self.world.batch_dim)
        ]

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
        magnitude_pos = self.interpolate(
            agent.action.u[:, 0], target_min=0, target_max=1
        )

        magnitude_neg = self.interpolate(
            agent.action.u[:, 1], target_min=0, target_max=1
        )

        magnitude = magnitude_pos - magnitude_neg

        # Set angle
        # turn_angle = torch.pi / 4

        # in_theta = self.interpolate(
        #     agent.action.u[:, 1], target_min=-turn_angle, target_max=turn_angle
        # )

        in_theta = 0

        # Get heading vector
        agent_rot = agent.state.rot % (2 * torch.pi)
        heading_offset = agent_rot + torch.pi / 2

        theta = (in_theta + heading_offset) % (2 * torch.pi)

        x = torch.cos(theta).squeeze(-1) * magnitude
        y = torch.sin(theta).squeeze(-1) * magnitude

        agent.state.force = torch.stack((x, y), dim=-1)

        # Unconstrained action
        # agent.state.force = agent.action.u[:] * 0.4

        # Join action
        # if agent.state.join.any():
        #     self.world.detach_joint(self.joint_list[0])

    def get_targets(self):
        return [landmark for landmark in self.world.landmarks if not landmark.is_joint]

    def get_position_tensors(self):
        targets = self.get_targets()
        agent_pos = []
        target_pos = []

        for agent, target in zip(self.world.agents, targets):
            agent_pos.append(agent.state.pos)
            target_pos.append(target.state.pos)

        agent_pos = torch.stack(agent_pos).transpose(1, 0)
        agent_centroids = agent_pos.mean(dim=1)
        target_pos = torch.stack(target_pos).transpose(1, 0)
        target_centroids = target_pos.mean(dim=1)

        return agent_pos, target_pos, agent_centroids, target_centroids

    def calculate_frechet_reward(self) -> torch.Tensor:

        agent_pos, target_pos, _, _ = self.get_position_tensors()

        f_dist = batch_discrete_frechet_distance(agent_pos, target_pos)
        f_rew = 1 / torch.exp(f_dist)

        return f_rew

    def calculate_centroid_reward(self) -> torch.Tensor:

        c_dist = []
        for n_env in range(self.world.batch_dim):
            c_dist.append(
                torch.norm(
                    self.target_chains[n_env].centroid
                    - self.agent_chains[n_env].centroid,
                )
            )
        c_dist = torch.stack(c_dist)
        c_rew = 1 / torch.exp(c_dist)

        return c_rew

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        if is_first:

            # Calculate G
            self.update_agent_chains()

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

            self.global_rew = self.frechet_rew + self.centroid_rew

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

    def agent_representation(self, agent: Agent, scope: str):

        agent_pos, target_pos, agent_centroids, target_centroids = (
            self.get_position_tensors()
        )

        a_chain_centroid_pos = agent_centroids

        t_chain_centroid = target_centroids

        # t_chain_orientation = self.target_chains[0].orientation

        # a_chain_orientation = self.agent_chains[0].orientation

        a_pos_rel_2_t_centroid = agent.state.pos - t_chain_centroid

        f_dist = batch_discrete_frechet_distance(agent_pos, target_pos)

        c_diff_vect = t_chain_centroid - a_chain_centroid_pos

        total_moment = 0
        total_force = 0

        positions = []
        vels = []
        ang_vels = []
        ang_pos = []

        for a in self.world.agents:
            r = a.state.pos - a_chain_centroid_pos
            total_moment += self.calculate_moment(r, a.state.force)
            total_force += a.state.force

            positions.append(a.state.pos)
            vels.append(a.state.vel)
            ang_vels.append(a.state.ang_vel)
            ang_pos.append(a.state.rot)

        positions = torch.stack(positions, dim=1)
        vels = torch.stack(vels, dim=1)
        ang_vels = torch.stack(ang_vels, dim=1)
        ang_pos = torch.stack(ang_pos, dim=1)

        a_chain_centroid_vel = vels.mean(dim=1)
        a_chain_centroid_ang_vel = ang_vels.mean(dim=1)
        a_chain_centroid_ang_pos = ang_pos.mean(dim=1) % (2 * torch.pi)

        # a_chain_orientation_rel_2_target = (
        #     t_chain_orientation - a_chain_orientation
        # ).unsqueeze(0)

        # Agent specific
        a_pos_rel_2_centroid = agent.state.pos - a_chain_centroid_pos
        a_ang_vel_rel_2_centroid = angular_velocity(
            a_pos_rel_2_centroid, agent.state.vel
        ).unsqueeze(0)

        # Complete observation
        match (scope):
            case "global":
                observation = torch.cat(
                    [
                        # Global state
                        a_chain_centroid_pos,
                        a_chain_centroid_vel,
                        a_chain_centroid_ang_pos / (2 * torch.pi),
                        a_chain_centroid_ang_vel,
                        total_force,
                        total_moment.unsqueeze(-1),
                        f_dist.unsqueeze(-1),
                        c_diff_vect,
                        # For IPPO actor
                        # a_pos_rel_2_centroid,
                        # a_pos_rel_2_t_centroid,
                        # agent.state.pos,
                        # agent.state.vel,
                        # agent.state.rot % (2 * torch.pi) / (2 * torch.pi),
                        # agent.state.ang_vel,
                    ],
                    dim=-1,
                ).float()
            case "global_plus_local":
                observation = torch.cat(
                    [
                        # Global state
                        a_chain_centroid_pos,
                        a_chain_centroid_vel,
                        a_chain_centroid_ang_pos / (2 * torch.pi),
                        a_chain_centroid_ang_vel,
                        total_force,
                        total_moment.unsqueeze(-1),
                        f_dist.unsqueeze(-1),
                        c_diff_vect,
                        # For IPPO actor
                        a_pos_rel_2_centroid,
                        a_pos_rel_2_t_centroid,
                        agent.state.pos,
                        agent.state.vel,
                        agent.state.rot % (2 * torch.pi) / (2 * torch.pi),
                        agent.state.ang_vel,
                    ],
                    dim=-1,
                )

            case "local":
                observation = torch.cat(
                    [
                        a_pos_rel_2_centroid,
                        a_pos_rel_2_t_centroid,
                        agent.state.pos,
                        agent.state.vel,
                        agent.state.rot % (2 * torch.pi) / (2 * torch.pi),
                        agent.state.ang_vel,
                    ],
                    dim=-1,
                )

        return observation

    def observation(self, agent: Agent):
        return self.agent_representation(agent, self.state_representation)

    def done(self):
        target_reached = self.total_rew > 0.98
        out_of_bounds = self.is_out_of_bounds()
        return torch.logical_or(target_reached, out_of_bounds)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "global_reward": (self.global_rew),
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []
        # Target ranges
        targets = self.get_targets()
        for target in targets:
            range_circle = rendering.make_circle(self.target_radius, filled=False)
            xform = rendering.Transform()
            xform.set_translation(*target.state.pos[env_index])
            range_circle.add_attr(xform)
            range_circle.set_color(*COLOR_MAP["RED"].value)
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
