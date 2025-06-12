#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import Dict, List

import torch
from torch import Tensor
from vmas.simulator.joints import Joint
from vmas.simulator.core import Agent, Landmark, Box, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils

from learning.environments.salp_navigate.dynamics import SalpDynamics
from learning.environments.salp_navigate.utils import (
    COLOR_LIST,
    COLOR_MAP,
    generate_target_points,
    generate_bending_curve,
    batch_discrete_frechet_distance,
    generate_random_coordinate_outside_box,
    rotate_points,
    calculate_moment,
    internal_angles_xy,
    wrap_to_pi,
    menger_curvature,
    centre_and_rotate,
    get_neighbor_angles,
    binary_encode,
)
from learning.environments.salp_navigate.types import GlobalObservation
import random
import math

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

torch.set_printoptions(precision=5)


class SalpNavigateDomain(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # CONSTANTS
        self.agent_radius = 0.02
        self.agent_joint_length = 0.06
        self.agent_max_angle = 45
        self.agent_min_angle = -45
        self.u_multiplier = 1.0
        self.target_radius = self.agent_radius / 2
        self.frechet_thresh = 0.95
        self.max_n_agents = 24
        self.min_n_agents = 8

        self.viewer_zoom = kwargs.pop("viewer_zoom", 1.45)

        # Agents
        self.n_agents = kwargs.pop("n_agents", self.min_n_agents)
        self.state_representation = kwargs.pop("state_representation", "local")
        self.agent_chains = [None for _ in range(batch_dim)]

        # Targets
        self.target_chains = [None for _ in range(batch_dim)]

        if kwargs.pop("shuffle_agents_positions", False):
            random.shuffle(self.agents_idx)

        # Check if we are training or evaluating
        self.training = kwargs.pop("training", True)

        # Environment
        self.x_semidim = (
            1.0
            if self.n_agents <= 8
            else 1.0 + self.n_agents / 2 * self.agent_joint_length
        )
        self.y_semidim = (
            1.0
            if self.n_agents <= 8
            else 1.0 + self.n_agents / 2 * self.agent_joint_length
        )

        if self.training:
            # Set a smaller world size for training like a fence
            self.world_x_dim = self.n_agents / 4
            self.world_y_dim = self.n_agents / 4
        else:
            # Increase world size for evaluation, like removing the fence
            self.world_x_dim = 10
            self.world_y_dim = 10

        # Reward Shaping
        self.frechet_shaping_factor = 1.0
        self.centroid_shaping_factor = 1.0
        self.curvature_shaping_factor = 1.0
        self.distance_shaping_factor = 1.0

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.device = device
        # Make world
        world = World(
            batch_dim=batch_dim,
            x_semidim=self.world_x_dim,
            y_semidim=self.world_y_dim,
            device=device,
            substeps=15,
            collision_force=1500,
            joint_force=900,
            contact_margin=1e-3,
            torque_constraint_force=0.1,
        )

        # Set targets
        self.targets = []
        for n_agent in range(self.n_agents):
            target = Landmark(
                name=f"target_{n_agent}_chain",
                shape=Sphere(radius=self.target_radius),
                color=COLOR_LIST[n_agent],
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
                color=COLOR_LIST[n_agent],
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
                rotate_a=False,
                rotate_b=False,
                collidable=False,
                width=0,
            )
            world.add_joint(joint)
            self.joints.append(joint)

        # Initialize reward tensors
        self.reached_goal_bonus = 1
        self.global_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.centroid_rew = self.global_rew.clone()
        self.frechet_rew = self.global_rew.clone()
        self.curvature_rew = self.global_rew.clone()
        self.distance_rew = self.global_rew.clone()

        # Initialize memory
        self.internal_angles_prev = torch.zeros(
            (batch_dim, self.n_agents - 2), device=device, dtype=torch.float32
        )  # n_agents-2 internal angles
        self.link_angles_prev = torch.zeros(
            (batch_dim, self.n_agents - 2), device=device, dtype=torch.float32
        )  # n_agents-1 link angles
        self.relative_angles_prev = torch.zeros(
            (batch_dim, self.n_agents, 2), device=device, dtype=torch.float32
        )  # n_agents-1 link angles

        world.zero_grad()

        return world

    def reset_world_at(self, env_index: int = None):
        joint_delta_x = self.agent_joint_length / 2
        joint_delta_y = 0.0
        agent_scale = 0.1
        agent_offset = 0.0

        target_offset = self.n_agents * self.agent_joint_length
        target_scale = self.n_agents * self.agent_joint_length

        # Rotation params
        agent_rotation_angles = [
            random.uniform(0, 2 * math.pi) for _ in range(self.world.batch_dim)
        ]
        agent_rotation_tensor = torch.tensor(
            agent_rotation_angles, device=self.device
        ).unsqueeze(-1)
        target_rotation_angle = random.uniform(0, 2 * math.pi)

        if env_index is None:
            # Create new agent and target chains
            self.agent_chains = [
                self.create_agent_chain(
                    offset=agent_offset,
                    scale=agent_scale,
                    theta_min=0.0,
                    theta_max=0.0,
                    rotation_angle=agent_rotation_tensor[i],
                )
                for i in range(self.world.batch_dim)
            ]

            self.target_chains = [
                self.create_target_chain(
                    offset=target_offset,
                    scale=target_scale,
                    rotation_angle=target_rotation_angle,
                )
                for _ in range(self.world.batch_dim)
            ]

            # Set agent positions and rotations
            agent_chain_tensor = torch.stack(
                [agent_chain for agent_chain in self.agent_chains]
            )
            for i, agent in enumerate(self.agents):
                pos = agent_chain_tensor[:, i, :]
                agent.set_pos(pos, batch_index=None)
                agent.set_rot(agent_rotation_tensor, batch_index=None)

            # Set target positions
            target_chain_tensor = torch.stack(
                [target_chain for target_chain in self.target_chains]
            )
            for i, target in enumerate(self.targets):
                pos = target_chain_tensor[:, i, :]
                target.set_pos(pos, batch_index=None)

            joint_delta = torch.tensor(
                (joint_delta_x, joint_delta_y), device=self.device
            ).repeat(self.world.batch_dim, 1)

            for i, joint in enumerate(self.joints):
                joint.landmark.set_pos(
                    self.agents[i].state.pos + joint_delta, batch_index=None
                )

            a_pos = self.get_agent_chain_position()
            self.internal_angles_prev, self.link_angles_prev = internal_angles_xy(a_pos)

            relative_angles = [
                get_neighbor_angles(a_pos, self.world.agents.index(a), self.n_agents)
                for a in self.world.agents
            ]

            self.relative_angles_prev = torch.stack(relative_angles).transpose(1, 0)

            t_pos = self.get_target_chain_position()

            f_dist, _ = self.calculate_frechet_reward(a_pos, t_pos)
            c_dist, _ = self.calculate_centroid_reward(
                a_pos.mean(dim=1), t_pos.mean(dim=1)
            )
            curvature = self.calculate_curvature_reward(a_pos, t_pos)
            dist_rew = self.calculate_distance_reward(a_pos, t_pos)

            self.frechet_shaping = f_dist * self.frechet_shaping_factor
            self.centroid_shaping = c_dist * self.centroid_shaping_factor
            self.curvature_shaping = curvature * self.curvature_shaping_factor
            self.distance_shaping = dist_rew * self.distance_shaping_factor

        else:
            self.agent_chains[env_index] = self.create_agent_chain(
                offset=agent_offset,
                scale=agent_scale,
                theta_min=0.0,
                theta_max=0.0,
                rotation_angle=agent_rotation_tensor[env_index],
            )
            self.target_chains[env_index] = self.create_target_chain(
                offset=target_offset,
                scale=target_scale,
                rotation_angle=target_rotation_angle,
            )

            for n_agent, agent in enumerate(self.world.agents):
                pos = self.agent_chains[env_index][n_agent]
                agent.set_pos(pos, batch_index=env_index)
                agent.set_rot(agent_rotation_tensor[env_index], batch_index=env_index)

            for n_target, target in enumerate(self.targets):
                pos = self.target_chains[env_index][n_target]
                target.set_pos(pos, batch_index=env_index)

            joint_delta = torch.tensor(
                (joint_delta_x, joint_delta_y), device=self.device
            )

            for i, joint in enumerate(self.joints):
                joint.landmark.set_pos(
                    self.agents[i].state.pos[env_index] + joint_delta,
                    batch_index=env_index,
                )

            a_pos = self.get_agent_chain_position()
            self.internal_angles_prev[env_index], self.link_angles_prev[env_index] = (
                internal_angles_xy(a_pos[env_index].unsqueeze(0))
            )

            relative_angles = [
                get_neighbor_angles(a_pos, self.world.agents.index(a), self.n_agents)
                for a in self.world.agents
            ]

            self.relative_angles_prev[env_index] = torch.stack(
                relative_angles
            ).transpose(1, 0)[env_index]

            t_pos = self.get_target_chain_position()

            f_dist, _ = self.calculate_frechet_reward(a_pos, t_pos)
            c_dist, _ = self.calculate_centroid_reward(
                a_pos.mean(dim=1), t_pos.mean(dim=1)
            )
            curvature = self.calculate_curvature_reward(a_pos, t_pos)
            dist_rew = self.calculate_distance_reward(a_pos, t_pos)

            self.frechet_shaping[env_index] = (
                f_dist[env_index] * self.frechet_shaping_factor
            )
            self.centroid_shaping[env_index] = (
                c_dist[env_index] * self.centroid_shaping_factor
            )
            self.curvature_shaping[env_index] = (
                curvature[env_index] * self.curvature_shaping_factor
            )
            self.distance_shaping[env_index] = (
                dist_rew[env_index] * self.distance_shaping_factor
            )

    def is_out_of_bounds(self, x_coord, y_coord):
        """Boolean mask of shape (n_envs,) – True if agent is out of bounds."""
        out_of_bounds = []

        for agent in self.agents:
            pos = agent.state.pos  # (n_envs, 2)
            x_ok = pos[..., 0].abs() <= x_coord - 1e-4
            y_ok = pos[..., 1].abs() <= y_coord - 1e-4
            out_of_bounds.append(~(x_ok & y_ok))

        out_of_bounds = torch.stack(out_of_bounds).transpose(1, 0).any(dim=-1)

        return out_of_bounds

    def create_agent_chain(
        self, offset, scale, theta_min, theta_max, rotation_angle: float = 0.0
    ):
        x_coord, y_coord = generate_random_coordinate_outside_box(
            offset,
            scale,
            1.0,
            1.0,
        )
        chain = rotate_points(
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
        ).to(self.device)
        return chain

    def create_target_chain(self, offset, scale, rotation_angle: float = 0.0):
        x_coord, y_coord = generate_random_coordinate_outside_box(
            offset,
            scale,
            1.0,
            1.0,
        )

        n_bends = random.choice([0, 1])
        radius = random.uniform(0.05, 0.3)
        radius_scaling = (
            self.n_agents // 3
        )  # 3 because it's the minimum amount of points for a curve

        chain = rotate_points(
            points=generate_bending_curve(
                x0=x_coord,
                y0=y_coord,
                n_points=self.n_agents,
                max_dist=self.agent_joint_length,
                radius=radius * radius_scaling,
                n_bends=n_bends,
            ),
            angle_rad=rotation_angle,
        ).to(self.device)

        return chain

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

    def get_agent_chain_position(self):
        agent_pos = [a.state.pos for a in self.world.agents]
        return torch.stack(agent_pos).transpose(1, 0).float()

    def get_target_chain_position(self):
        targets = self.get_targets()
        target_pos = [t.state.pos for t in targets]
        return torch.stack(target_pos).transpose(1, 0).float()

    def calculate_distance_reward(self, a_pos: torch.Tensor, t_pos: torch.Tensor):
        pos_err = t_pos - a_pos  # [B, N, 2]
        dists = torch.linalg.norm(pos_err, dim=-1)
        dist_rew = -dists.mean(dim=-1)
        return dist_rew

    def calculate_frechet_reward(
        self, a_pos: torch.Tensor, t_pos: torch.Tensor, aligned: bool = False
    ) -> torch.Tensor:

        if aligned:
            a_pos, t_pos = centre_and_rotate(a_pos, t_pos)

        f_dist = batch_discrete_frechet_distance(a_pos, t_pos)
        f_rew = 1 / torch.exp(f_dist)

        return -f_dist, f_rew

    def calculate_centroid_reward(
        self, a_centroid: torch.Tensor, t_centroid: torch.Tensor
    ) -> torch.Tensor:

        c_dist = torch.norm(a_centroid - t_centroid, dim=1)
        c_rew = 1 / torch.exp(c_dist)

        return -c_dist, c_rew

    def calculate_curvature_reward(
        self, a_pos: torch.Tensor, t_pos: torch.Tensor
    ) -> torch.Tensor:

        k = menger_curvature(a_pos, self.agent_joint_length)
        k_star = menger_curvature(t_pos, self.agent_joint_length)

        rew = -torch.sum(torch.abs(k - k_star), dim=-1)

        return rew

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:

            # Calculate G

            self.frechet_rew[:] = 0
            self.centroid_rew[:] = 0
            self.curvature_rew[:] = 0
            self.distance_rew[:] = 0

            # Get chain positions
            agent_pos = self.get_agent_chain_position()
            target_pos = self.get_target_chain_position()

            # Distance reward
            dist_rew = self.calculate_distance_reward(agent_pos, target_pos)
            dist_shaping = dist_rew * self.distance_shaping_factor
            self.distance_rew = dist_shaping - self.distance_shaping
            self.distance_shaping = dist_shaping

            # Frechet reward

            _, f_rew = self.calculate_frechet_reward(agent_pos, target_pos)
            # f_dist, _ = self.calculate_frechet_reward(
            #     agent_pos, target_pos, aligned=True
            # )
            # frechet_shaping = f_dist * self.frechet_shaping_factor
            # self.frechet_rew += frechet_shaping - self.frechet_shaping
            # self.frechet_shaping = frechet_shaping

            # Curvature reward
            # curvature_rew = self.calculate_curvature_reward(agent_pos, target_pos)
            # curvature_shaping = curvature_rew * self.curvature_shaping_factor
            # self.curvature_rew += torch.tanh(curvature_shaping - self.curvature_shaping)
            # self.curvature_shaping = curvature_shaping

            # Centroid reward
            # agent_centroid = agent_pos.mean(dim=1)
            # target_centroid = target_pos.mean(dim=1)
            # cent_dist, _ = self.calculate_centroid_reward(
            #     agent_centroid, target_centroid
            # )
            # centroid_shaping = cent_dist * self.centroid_shaping_factor
            # self.centroid_rew += centroid_shaping - self.centroid_shaping
            # self.centroid_shaping = centroid_shaping

            # Get reward for reaching the goal
            self.total_rew = f_rew
            goal_reached_rew = torch.zeros(
                self.world.batch_dim, device=self.device, dtype=torch.float32
            )
            goal_reached_mask = self.total_rew > self.frechet_thresh
            goal_reached_rew += self.reached_goal_bonus * goal_reached_mask.int()

            # Mix all rewards
            self.global_rew = self.distance_rew if self.training else f_rew

        return self.global_rew

    def get_heading(self, agent: Agent):
        x = torch.cos(agent.state.rot + 1.5 * torch.pi).squeeze(-1)
        y = torch.sin(agent.state.rot + 1.5 * torch.pi).squeeze(-1)

        return torch.stack((x, y), dim=-1)

    def agent_representation(self, agent: Agent, scope: str):

        # Agent specific
        a_pos_rel_2_t_centroid = (
            agent.state.pos - self.global_observation.t_chain_centroid_pos
        )

        a_vel_rel_2_centroid = (
            agent.state.vel - self.global_observation.a_chain_centroid_vel
        )

        a_pos_rel_2_centroid = (
            agent.state.pos - self.global_observation.a_chain_centroid_pos
        )

        # Complete observation
        match (scope):
            case "global":
                observation = torch.cat(
                    [
                        # Shape features
                        torch.sin(self.global_observation.a_chain_internal_angles),
                        torch.cos(self.global_observation.a_chain_internal_angles),
                        self.global_observation.a_chain_internal_angles_speed,
                        self.global_observation.curvature,
                        # Whole body motion
                        self.global_observation.a_chain_centroid_pos,
                        self.global_observation.a_chain_centroid_vel,
                        self.global_observation.a_chain_centroid_ang_pos,
                        self.global_observation.a_chain_centroid_ang_vel,
                        self.global_observation.total_force,
                        self.global_observation.total_moment,
                        # Target features
                        self.global_observation.frechet_dist,
                        self.global_observation.t_chain_centroid_pos
                        - self.global_observation.a_chain_centroid_pos,
                    ],
                    dim=-1,
                ).float()

            case "local":
                # Get agent information
                is_first = agent == self.world.agents[0]
                is_last = agent == self.world.agents[-1]

                idx = self.world.agents.index(agent)

                # Encode agent id
                encoding_len = 6
                encoded_idx = torch.zeros(
                    (self.world.batch_dim, encoding_len),
                    dtype=torch.float32,
                    device=self.device,
                ) + binary_encode(idx, encoding_len)

                # Get neighbor forces
                neighbor_forces = torch.zeros(
                    (self.world.batch_dim, 4), dtype=torch.float32, device=self.device
                )
                if is_first:
                    neighbor_forces[:, 2:] = self.global_observation.a_chain_all_forces[
                        :, 1, :
                    ]
                elif is_last:
                    neighbor_forces[:, :2] = self.global_observation.a_chain_all_forces[
                        :, -2, :
                    ]
                else:
                    neighbor_forces = self.global_observation.a_chain_all_forces[
                        :, idx - 1 : idx + 2 : 2, :
                    ].flatten(start_dim=1)

                # Get distance to assigned position
                a_pos_2_t_pos_err = (
                    self.global_observation.t_chain_all_pos[:, idx, :]
                    - self.global_observation.a_chain_all_pos[:, idx, :]
                )

                observation = torch.cat(
                    [
                        # Agent id
                        encoded_idx,
                        # Neighbor data
                        torch.sin(
                            self.global_observation.a_chain_relative_angles[:, idx, :]
                        ),
                        torch.cos(
                            self.global_observation.a_chain_relative_angles[:, idx, :]
                        ),
                        self.global_observation.a_chain_relative_angles_speed[
                            :, idx, :
                        ],
                        neighbor_forces,
                        # Local data
                        a_pos_rel_2_centroid,
                        agent.state.pos,
                        agent.state.vel,
                        wrap_to_pi(agent.state.rot),
                        agent.state.ang_vel,
                        # Target data
                        a_pos_rel_2_t_centroid,
                        a_vel_rel_2_centroid,
                        a_pos_2_t_pos_err,
                    ],
                    dim=-1,
                ).float()

        return observation

    def observation(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        if is_first:
            # Calculate global state
            agent_pos = self.get_agent_chain_position()
            target_pos = self.get_target_chain_position()
            a_chain_centroid_pos = agent_pos.mean(dim=1)
            t_chain_centroid_pos = target_pos.mean(dim=1)

            aligned_agent_pos = agent_pos - agent_pos.mean(dim=1, keepdim=True)
            aligned_target_pos = target_pos - target_pos.mean(dim=1, keepdim=True)

            total_moment = 0

            vels = []
            ang_vels = []
            ang_pos = []
            forces = []
            relative_angles = []

            for a in self.world.agents:

                r = a.state.pos - a_chain_centroid_pos
                total_moment += calculate_moment(r, a.state.force)

                vels.append(a.state.vel)
                ang_vels.append(a.state.ang_vel)
                ang_pos.append(a.state.rot)
                forces.append(a.state.force)
                relative_angles.append(
                    get_neighbor_angles(
                        agent_pos, self.world.agents.index(a), self.n_agents
                    )
                )

            vels = torch.stack(vels).transpose(1, 0)
            ang_vels = torch.stack(ang_vels).transpose(1, 0)
            ang_pos = torch.stack(ang_pos).transpose(1, 0)
            forces = torch.stack(forces).transpose(1, 0)
            relative_angles = torch.stack(relative_angles).transpose(1, 0)

            internal_angles, link_angles = internal_angles_xy(agent_pos)

            # Calculate angle derivatives
            internal_angles_speed = (
                wrap_to_pi(internal_angles - self.internal_angles_prev) / self.world.dt
            )

            link_angles_speed = (
                wrap_to_pi(link_angles - self.link_angles_prev) / self.world.dt
            )

            relative_angles_speed = (
                wrap_to_pi(relative_angles - self.relative_angles_prev) / self.world.dt
            )

            # Store previous dtheta
            self.internal_angles_prev = internal_angles.clone()
            self.link_angles_prev = link_angles.clone()
            self.relative_angles_prev = relative_angles.clone()

            # Build global observation
            self.global_observation = GlobalObservation(
                # Menger curvature
                menger_curvature(agent_pos, self.agent_joint_length)
                - menger_curvature(target_pos, self.agent_joint_length),
                # Internal angle data
                internal_angles,
                internal_angles_speed,
                # Link angles
                link_angles,
                link_angles_speed,
                # Relative angles
                relative_angles,
                relative_angles_speed,
                # Raw obs
                target_pos,
                agent_pos,
                vels.flatten(start_dim=1),
                ang_pos.flatten(start_dim=1),
                ang_vels.flatten(start_dim=1),
                forces,
                # Condensed obs
                t_chain_centroid_pos,
                a_chain_centroid_pos,
                vels.mean(dim=1),
                wrap_to_pi(ang_pos.mean(dim=1)),
                ang_vels.mean(dim=1),
                forces.sum(dim=1),
                total_moment.unsqueeze(-1),
                batch_discrete_frechet_distance(
                    aligned_agent_pos, aligned_target_pos
                ).unsqueeze(-1),
            )

            # print("\n")

            # print(f"dtheta {dtheta[0]}")
            # print(f"sin_dtheta {self.global_observation.a_chain_sin_dtheta[0]}")
            # print(f"cos_dtheta {self.global_observation.a_chain_cos_dtheta[0]}")
            # print(f"bend_speed {self.global_observation.a_chain_bend_speed[0]}")

            # print(
            #     f"centroid_ang_pos {self.global_observation.a_chain_centroid_ang_pos}"
            # )
            # print(
            #     f"centroid_ang_vel {self.global_observation.a_chain_centroid_ang_vel}"
            # )
            # print(f"total_force {self.global_observation.total_force}")
            # print(f"total_moment {self.global_observation.total_moment}")

        return self.agent_representation(agent, self.state_representation)

    def done(self):
        target_reached = self.total_rew > self.frechet_thresh
        out_of_bounds = self.is_out_of_bounds(
            self.world.x_semidim, self.world.y_semidim
        )
        return torch.logical_or(target_reached, out_of_bounds)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        chain_pos = self.get_agent_chain_position()
        target_pos = self.get_target_chain_position()
        return {"target_pose": (target_pos), "chain_pose": (chain_pos)}

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []
        # Target ranges
        targets = self.get_targets()
        for i, target in enumerate(targets):
            range_circle = rendering.make_circle(self.target_radius, filled=False)
            xform = rendering.Transform()
            xform.set_translation(*target.state.pos[env_index])
            range_circle.add_attr(xform)
            range_circle.set_color(*COLOR_LIST[i])
            geoms.append(range_circle)

        a_pos = self.get_agent_chain_position()

        range_circle = rendering.make_circle(self.target_radius, filled=False)
        xform = rendering.Transform()
        xform.set_translation(*a_pos[env_index].mean(dim=0))
        range_circle.add_attr(xform)
        range_circle.set_color(*COLOR_MAP["BLACK"].value)
        geoms.append(range_circle)

        t_pos = self.get_target_chain_position()

        range_circle = rendering.make_circle(self.target_radius, filled=False)
        xform = rendering.Transform()
        xform.set_translation(*t_pos[env_index].mean(dim=0))
        range_circle.add_attr(xform)
        range_circle.set_color(*COLOR_MAP["BLACK"].value)
        geoms.append(range_circle)

        return geoms
