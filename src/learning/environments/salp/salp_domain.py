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
    generate_bending_curve,
    batch_discrete_frechet_distance,
    angular_velocity,
    generate_random_coordinate_outside_box,
    rotate_points,
    calculate_moment,
    internal_angles_xy,
    bending_speed,
    wrap_to_pi,
    menger_curvature,
    centre_and_rotate,
)
from learning.environments.salp.types import Chain, GlobalObservation
import random
import math
from copy import deepcopy

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

torch.set_printoptions(precision=2)


class SalpDomain(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # CONSTANTS
        self.agent_radius = 0.02
        self.agent_joint_length = 0.06
        self.agent_max_angle = 45
        self.agent_min_angle = -45
        self.u_multiplier = 1.0
        self.target_radius = self.agent_radius / 2
        self.frechet_thresh = 0.98

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

        # Reward Shaping
        self.frechet_shaping_factor = 1.0
        self.centroid_shaping_factor = 1.0
        self.curvature_shaping_factor = 0.1

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.device = device
        # Make world
        world = SalpWorld(
            batch_dim=batch_dim,
            x_semidim=self.x_semidim * 1.2,
            y_semidim=self.y_semidim * 1.2,
            device=device,
            substeps=15,
            collision_force=1500,
            joint_force=900,
            contact_margin=1e-3,
            torque_constraint_force=0.1,
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
                rotate_a=False,
                rotate_b=False,
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
        self.curvature_rew = self.global_rew.clone()

        # Initialize memory
        self.dtheta_prev = torch.zeros(
            (batch_dim, self.n_agents - 2), device=device, dtype=torch.float32
        )  # n_agents-2 links

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
                self.create_agent_chain(
                    offset=agent_offset, scale=agent_scale, theta_min=0.0, theta_max=0.0
                )
                for _ in range(self.world.batch_dim)
            ]

            self.target_chains = [
                self.create_target_chain(
                    offset=target_offset,
                    scale=target_scale,
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

            a_pos = self.get_agent_chain_position()
            self.dtheta_prev = internal_angles_xy(a_pos)

            t_pos = self.get_target_chain_position()

            self.frechet_shaping = (
                self.calculate_frechet_reward(a_pos, t_pos)
                * self.frechet_shaping_factor
            )
            self.centroid_shaping = (
                self.calculate_centroid_reward(a_pos.mean(dim=1), t_pos.mean(dim=1))
                * self.centroid_shaping_factor
            )
            self.curvature_shaping = (
                self.calculate_curvature_reward(a_pos, t_pos)
                * self.curvature_shaping_factor
            )

        else:
            self.agent_chains[env_index] = self.create_agent_chain(
                offset=agent_offset, scale=agent_scale, theta_min=0.0, theta_max=0.0
            )
            self.target_chains[env_index] = self.create_target_chain(
                offset=target_offset,
                scale=target_scale,
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

            a_pos = self.get_agent_chain_position()
            self.dtheta_prev[env_index] = internal_angles_xy(
                a_pos[env_index].unsqueeze(0)
            )

            t_pos = self.get_target_chain_position()

            self.frechet_shaping[env_index] = (
                self.calculate_frechet_reward(a_pos, t_pos)[env_index]
                * self.frechet_shaping_factor
            )
            self.centroid_shaping[env_index] = (
                self.calculate_centroid_reward(a_pos.mean(dim=1), t_pos.mean(dim=1))[
                    env_index
                ]
                * self.centroid_shaping_factor
            )
            self.curvature_shaping[env_index] = (
                self.calculate_curvature_reward(a_pos, t_pos)[env_index]
                * self.curvature_shaping_factor
            )

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

    def create_agent_chain(
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
            ).to(self.device),
        )
        return chain

    def create_target_chain(self, offset, scale, rotation_angle: float = 0.0):
        x_coord, y_coord = generate_random_coordinate_outside_box(
            offset,
            scale,
            self.x_semidim,
            self.y_semidim,
        )

        n_bends = random.choice([0, 1])
        radius = random.uniform(0.05, 0.3)
        radius_scaling = (
            self.n_agents // 3
        )  # 3 because it's the minimum amount of points for a curve

        chain = Chain(
            path=rotate_points(
                points=generate_bending_curve(
                    x0=x_coord,
                    y0=y_coord,
                    n_points=self.n_agents,
                    max_dist=self.agent_joint_length,
                    radius=radius * radius_scaling,
                    n_bends=n_bends,
                ),
                angle_rad=rotation_angle,
            ).to(self.device),
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

    def get_agent_chain_position(self):
        agent_pos = [a.state.pos for a in self.world.agents]
        return torch.stack(agent_pos).transpose(1, 0)

    def get_target_chain_position(self):
        targets = self.get_targets()
        target_pos = [t.state.pos for t in targets]
        return torch.stack(target_pos).transpose(1, 0)

    def calculate_frechet_reward(
        self, a_pos: torch.Tensor, t_pos: torch.Tensor, aligned: bool = False
    ) -> torch.Tensor:

        if aligned:
            a_pos, t_pos = centre_and_rotate(a_pos, t_pos)
        f_dist = batch_discrete_frechet_distance(a_pos, t_pos)
        f_rew = 1 / torch.exp(f_dist)

        return f_rew

    def calculate_centroid_reward(
        self, a_centroid: torch.Tensor, t_centroid: torch.Tensor
    ) -> torch.Tensor:

        c_dist = torch.norm(a_centroid - t_centroid, dim=1)
        c_rew = 1 / torch.exp(c_dist)

        return c_rew

    def calculate_curvature_reward(
        self, a_pos: torch.Tensor, t_pos: torch.Tensor, lambda_k: float = 0.5
    ) -> torch.Tensor:

        k = menger_curvature(a_pos)
        k_star = menger_curvature(t_pos)

        rew = -torch.sum(torch.abs(k - k_star), dim=-1)

        return rew

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:

            # Calculate G
            agent_pos = self.get_agent_chain_position()
            target_pos = self.get_target_chain_position()

            agent_centroid = agent_pos.mean(dim=1)
            target_centroid = target_pos.mean(dim=1)

            self.frechet_rew[:] = 0
            self.centroid_rew[:] = 0
            self.curvature_rew[:] = 0

            # Calculate shaping terms
            f_rew = self.calculate_frechet_reward(agent_pos, target_pos)
            frechet_shaping = f_rew * self.frechet_shaping_factor
            self.frechet_rew += frechet_shaping - self.frechet_shaping
            self.frechet_shaping = frechet_shaping

            # cent_rew = self.calculate_centroid_reward(agent_centroid, target_centroid)
            # centroid_shaping = cent_rew * self.centroid_shaping_factor
            # self.centroid_rew += centroid_shaping - self.centroid_shaping
            # self.centroid_shaping = centroid_shaping

            # curvature_rew = self.calculate_curvature_reward(agent_pos, target_pos)
            # curvature_shaping = curvature_rew * self.curvature_shaping_factor
            # self.curvature_rew += curvature_shaping - self.curvature_shaping
            # self.curvature_shaping = curvature_shaping

            self.total_rew = f_rew

            # Get reward for reaching the goal
            goal_reached_rew = torch.zeros(
                self.world.batch_dim, device=self.device, dtype=torch.float32
            )
            goal_reached_mask = self.total_rew > self.frechet_thresh
            goal_reached_rew += 10 * goal_reached_mask.int()

            # Mix all rewards
            self.global_rew = (
                self.frechet_rew * 10 + goal_reached_rew  # + self.centroid_rew
            )  # + self.curvature_rew

        rew = torch.cat([self.global_rew])

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

    def agent_representation(self, agent: Agent, scope: str):

        # a_chain_orientation_rel_2_target = (
        #     t_chain_orientation - a_chain_orientation
        # ).unsqueeze(0)

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
        a_ang_vel_rel_2_centroid = angular_velocity(
            a_pos_rel_2_centroid, agent.state.vel
        ).unsqueeze(0)

        # Complete observation
        match (scope):
            case "global":
                observation = torch.cat(
                    [
                        # Shape features
                        self.global_observation.a_chain_sin_dtheta,
                        self.global_observation.a_chain_cos_dtheta,
                        self.global_observation.a_chain_bend_speed,
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
                        # Condensed state
                        # self.global_observation.a_chain_centroid_pos,
                        # self.global_observation.a_chain_centroid_vel,
                        # self.global_observation.a_chain_centroid_ang_pos
                        # / (2 * torch.pi),
                        # self.global_observation.a_chain_centroid_ang_vel,
                        # self.global_observation.total_force,
                        # self.global_observation.total_moment.unsqueeze(-1),
                        # self.global_observation.frechet_dist.unsqueeze(-1),
                        # Raw state
                        # self.global_observation.a_chain_all_pos,
                        # self.global_observation.a_chain_all_vel,
                        # self.global_observation.a_chain_all_ang_pos / (2 * torch.pi),
                        # self.global_observation.a_chain_all_ang_vel,
                        # self.global_observation.total_force,
                        # self.global_observation.total_moment.unsqueeze(-1),
                        # self.global_observation.frechet_dist.unsqueeze(-1),
                    ],
                    dim=-1,
                ).float()
            case "global_plus_local":
                observation = torch.cat(
                    [
                        # Shape features
                        self.global_observation.a_chain_sin_dtheta,
                        self.global_observation.a_chain_cos_dtheta,
                        self.global_observation.a_chain_bend_speed,
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
                        # For IPPO actor
                        a_pos_rel_2_centroid,
                        a_vel_rel_2_centroid,
                        agent.state.pos,
                        agent.state.vel,
                        wrap_to_pi(agent.state.rot),
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
                        wrap_to_pi(agent.state.rot),
                        agent.state.ang_vel,
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
            total_force = 0

            vels = []
            ang_vels = []
            ang_pos = []

            for a in self.world.agents:
                r = a.state.pos - a_chain_centroid_pos
                total_moment += calculate_moment(r, a.state.force)
                total_force += a.state.force

                vels.append(a.state.vel)
                ang_vels.append(a.state.ang_vel)
                ang_pos.append(a.state.rot)

            vels = torch.stack(vels).transpose(1, 0)
            ang_vels = torch.stack(ang_vels).transpose(1, 0)
            ang_pos = torch.stack(ang_pos).transpose(1, 0)

            dtheta = internal_angles_xy(agent_pos)
            bend_speed = bending_speed(dtheta, self.dtheta_prev, dt=self.world.dt)

            # Store previous dtheta
            self.dtheta_prev = dtheta.clone()

            # Build global observation
            self.global_observation = GlobalObservation(
                # New obs
                torch.sin(dtheta),
                torch.cos(dtheta),
                bend_speed,
                # Raw obs
                target_pos.flatten(start_dim=1),
                agent_pos.flatten(start_dim=1),
                vels.flatten(start_dim=1),
                ang_pos.flatten(start_dim=1),
                ang_vels.flatten(start_dim=1),
                # Condensed obs
                t_chain_centroid_pos,
                a_chain_centroid_pos,
                vels.mean(dim=1),
                wrap_to_pi(ang_pos.mean(dim=1)),
                ang_vels.mean(dim=1),
                total_force,
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
        out_of_bounds = self.is_out_of_bounds()
        return torch.logical_or(target_reached, out_of_bounds)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "global_reward": (self.global_rew),
            "global_state": torch.cat(
                [
                    # Shape features
                    self.global_observation.a_chain_sin_dtheta,
                    self.global_observation.a_chain_cos_dtheta,
                    self.global_observation.a_chain_bend_speed,
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
            ).float(),
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
