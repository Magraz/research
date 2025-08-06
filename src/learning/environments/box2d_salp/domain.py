import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from Box2D import (
    b2World,
    b2PolygonShape,
    b2FixtureDef,
    b2RevoluteJointDef,
    b2CircleShape,
)

from learning.environments.box2d_salp.utils import COLORS_LIST

AGENT_CATEGORY = 0x0001  # Binary: 0001
BOUNDARY_CATEGORY = 0x0002  # Binary: 0010


class SalpChainEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, n_agents=8):
        super().__init__()

        self.n_agents = n_agents
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_agents, 2), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_agents, 4), dtype=np.float32
        )

        self.world = b2World(gravity=(0, 0))
        self.time_step = 1.0 / 60.0
        self.step_count = 0

        self.agents = []
        self.joints = []

        # Boundary parameters (customize as needed)
        self.world_width = 40
        self.world_height = 30
        self.boundary_thickness = 0.5

        # Pygame rendering setup
        self.screen = None
        self.clock = None
        self.screen_size = (800, 600)
        self.scale = 20.0  # Pixels per Box2D meter

        # Create boundary and agents
        self._create_boundary(
            self.world_width, self.world_height, self.boundary_thickness
        )
        self._create_chain()

        # Add force tracking
        self.applied_forces = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.force_scale = 2.0  # Scale factor for visualizing forces

    def _create_chain(self):
        self.agents.clear()
        self.joints.clear()

        previous_body = None
        for i in range(self.n_agents):
            fixture_def = b2FixtureDef(
                # shape=b2PolygonShape(box=(0.3, 0.5)),
                shape=b2CircleShape(radius=0.4),
                density=1.0,
                friction=0.3,
                isSensor=False,
            )

            fixture_def.filter.categoryBits = AGENT_CATEGORY
            fixture_def.filter.maskBits = (
                AGENT_CATEGORY
                | BOUNDARY_CATEGORY  # allow collisions between agents and boundaries
            )

            body = self.world.CreateDynamicBody(
                position=(10 + i * 1.5, 10),
                fixtures=fixture_def,
            )

            self.agents.append(body)

            if previous_body:
                joint = self.world.CreateJoint(
                    b2RevoluteJointDef(
                        bodyA=previous_body,
                        bodyB=body,
                        anchor=previous_body.worldCenter + (0.75, 0),
                        collideConnected=True,
                    )
                )
                self.joints.append(joint)
            previous_body = body

    def step(self, actions):
        for idx, agent in enumerate(self.agents):
            force_x = float(actions[idx][0]) * 10.0  # X component
            force_y = float(actions[idx][1]) * 10.0  # Y component

            # Store the 2D force vector for visualization
            self.applied_forces[idx] = [force_x, force_y]

            # Apply 2D force to agent
            agent.ApplyForceToCenter((force_x, force_y), True)

        self.world.Step(self.time_step, 6, 2)

        # For testing joints
        if self.step_count == 50:
            self._break_joint(self.joints[(self.n_agents // 2) - 1])

        if self.step_count > 110:
            self._join_on_proximity()

        # The observation
        obs = np.array(
            [
                [a.position.x, a.position.y, a.linearVelocity.x, a.linearVelocity.y]
                for a in self.agents
            ],
            dtype=np.float32,
        )

        reward = -np.linalg.norm(obs[:, :2] - np.array([10, 10]))
        terminated, truncated = False, False

        self.step_count += 1

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        for body in self.agents:
            self.world.DestroyBody(body)

        self._create_chain()

        obs = np.array(
            [
                [a.position.x, a.position.y, a.linearVelocity.x, a.linearVelocity.y]
                for a in self.agents
            ],
            dtype=np.float32,
        )

        if self.render_mode == "human":
            self.render()

        return obs, {}

    def _render_agents_as_circles(self):
        for idx, body in enumerate(self.agents):
            # Get circle position and radius
            center_x = body.position.x * self.scale
            center_y = self.screen_size[1] - body.position.y * self.scale
            radius = body.fixtures[0].shape.radius * self.scale  # Get radius from shape

            # Draw filled circle
            pygame.draw.circle(
                self.screen,
                COLORS_LIST[idx % len(COLORS_LIST)],
                (int(center_x), int(center_y)),
                int(radius),
            )

            # Draw circle outline for better visibility
            pygame.draw.circle(
                self.screen,
                (0, 0, 0),
                (int(center_x), int(center_y)),
                int(radius),
                2,  # Outline thickness
            )

    def _render_agents_as_boxes(self):
        for idx, body in enumerate(self.agents):
            for fixture in body.fixtures:
                shape = fixture.shape
                vertices = [(body.transform * v) * self.scale for v in shape.vertices]
                vertices = [(v[0], self.screen_size[1] - v[1]) for v in vertices]

                pygame.draw.polygon(self.screen, COLORS_LIST[idx], vertices)

    def render(self):
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Salp Chain Simulation")
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        # Draw boundary walls correctly positioned
        self._draw_boundary_walls()

        # Draw agents
        self._render_agents_as_circles()

        # Draw joints accurately using anchor points
        for joint in self.joints:
            anchor_a = joint.anchorA * self.scale
            anchor_b = joint.anchorB * self.scale

            # Adjust for pygame's inverted y-axis
            p1 = (anchor_a[0], self.screen_size[1] - anchor_a[1])
            p2 = (anchor_b[0], self.screen_size[1] - anchor_b[1])

            # Draw the joint line (pivot-to-pivot)
            pygame.draw.line(self.screen, (0, 0, 0), p1, p2, width=3)

            # Optionally, draw pivot points explicitly
            pygame.draw.circle(self.screen, (255, 0, 0), p1, radius=5)  # pivot on bodyA
            pygame.draw.circle(self.screen, (0, 0, 255), p2, radius=5)  # pivot on bodyB

        # Draw force vectors
        self._draw_force_vectors()

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _create_joint(self, bodyA, bodyB, anchor):
        joint_def = b2RevoluteJointDef(
            bodyA=bodyA, bodyB=bodyB, anchor=anchor, collideConnected=False
        )
        joint = self.world.CreateJoint(joint_def)
        self.joints.append(joint)
        return joint

    def _break_joint(self, joint):
        self.world.DestroyJoint(joint)
        self.joints.remove(joint)

    def _break_on_reaction_force(self):
        # Example logic to break/create joints dynamically
        for joint in self.joints[:]:
            reaction_force = joint.GetReactionForce(1.0 / self.time_step)
            reaction_force_mag = reaction_force.length

            # Break joint if force exceeds threshold
            if reaction_force_mag > 50.0:
                self._break_joint(joint)

    def _join_on_proximity(self):
        for i, bodyA in enumerate(self.agents):
            for bodyB in self.agents[i + 1 :]:
                dist = (bodyA.position - bodyB.position).length
                if dist < 1.0 and not self._bodies_connected(bodyA, bodyB):
                    anchor = (bodyA.position + bodyB.position) / 2
                    self._create_joint(bodyA, bodyB, anchor)

    def _bodies_connected(self, bodyA, bodyB):
        # Check if bodies are already connected
        for joint in self.joints:
            if (joint.bodyA == bodyA and joint.bodyB == bodyB) or (
                joint.bodyA == bodyB and joint.bodyB == bodyA
            ):
                return True
        return False

    def _create_boundary(self, width, height, thickness):
        """Create boundary walls that agents can collide with"""

        # Bottom wall
        bottom_wall = self.world.CreateStaticBody(
            position=(width / 2, thickness / 2),
            shapes=b2PolygonShape(box=(width / 2, thickness / 2)),
        )
        bottom_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        bottom_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY

        # Top wall
        top_wall = self.world.CreateStaticBody(
            position=(width / 2, height - thickness / 2),
            shapes=b2PolygonShape(box=(width / 2, thickness / 2)),
        )
        top_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        top_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY

        # Left wall
        left_wall = self.world.CreateStaticBody(
            position=(thickness / 2, height / 2),
            shapes=b2PolygonShape(box=(thickness / 2, height / 2)),
        )
        left_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        left_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY

        # Right wall
        right_wall = self.world.CreateStaticBody(
            position=(width - thickness / 2, height / 2),
            shapes=b2PolygonShape(box=(thickness / 2, height / 2)),
        )
        right_wall.fixtures[0].filterData.categoryBits = BOUNDARY_CATEGORY
        right_wall.fixtures[0].filterData.maskBits = AGENT_CATEGORY

    def _draw_boundary_walls(self):
        """Draw the actual boundary walls at their Box2D positions"""
        thickness = self.boundary_thickness

        # Bottom wall
        bottom_rect = pygame.Rect(
            0,  # Left edge
            self.screen_size[1] - thickness * self.scale,  # Bottom of screen
            self.world_width * self.scale,  # Full width
            thickness * self.scale,  # Wall thickness
        )
        pygame.draw.rect(self.screen, (0, 0, 0), bottom_rect)

        # Top wall
        top_rect = pygame.Rect(
            0,  # Left edge
            self.screen_size[1] - self.world_height * self.scale,  # Top position
            self.world_width * self.scale,  # Full width
            thickness * self.scale,  # Wall thickness
        )
        pygame.draw.rect(self.screen, (0, 0, 0), top_rect)

        # Left wall
        left_rect = pygame.Rect(
            0,  # Left edge of screen
            self.screen_size[1] - self.world_height * self.scale,  # Top position
            thickness * self.scale,  # Wall thickness
            self.world_height * self.scale,  # Full height
        )
        pygame.draw.rect(self.screen, (0, 0, 0), left_rect)

        # Right wall
        right_rect = pygame.Rect(
            self.world_width * self.scale - thickness * self.scale,  # Right position
            self.screen_size[1] - self.world_height * self.scale,  # Top position
            thickness * self.scale,  # Wall thickness
            self.world_height * self.scale,  # Full height
        )
        pygame.draw.rect(self.screen, (0, 0, 0), right_rect)

    def _draw_force_vectors(self):
        """Draw force vectors for each agent with enhanced 2D visualization"""
        for idx, (body, force) in enumerate(zip(self.agents, self.applied_forces)):
            # Get agent center position in screen coordinates
            center_x = body.position.x * self.scale
            center_y = self.screen_size[1] - body.position.y * self.scale

            # Calculate force vector magnitude
            force_magnitude = np.linalg.norm(force)
            if force_magnitude > 0.1:  # Only draw if force is significant
                # Scale the force vector for visibility
                scaled_force = force * self.force_scale

                end_x = center_x + scaled_force[0]
                end_y = center_y - scaled_force[1]  # Flip Y for screen coordinates

                # Draw force vector as arrow
                start_pos = (int(center_x), int(center_y))
                end_pos = (int(end_x), int(end_y))

                # Use thicker line for stronger forces
                line_width = max(1, int(force_magnitude * 0.5))

                # Draw main force line (thicker, colored by agent)
                pygame.draw.line(
                    self.screen,
                    COLORS_LIST[idx % len(COLORS_LIST)],
                    start_pos,
                    end_pos,
                    line_width,
                )

                # Draw arrowhead
                self._draw_arrowhead(
                    start_pos, end_pos, COLORS_LIST[idx % len(COLORS_LIST)]
                )

    def _draw_arrowhead(self, start_pos, end_pos, color):
        """Draw an arrowhead at the end of a force vector"""
        if start_pos == end_pos:
            return

        # Calculate arrow direction
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        length = np.sqrt(dx * dx + dy * dy)

        if length < 5:  # Don't draw tiny arrows
            return

        # Normalize direction
        dx /= length
        dy /= length

        # Arrow parameters
        arrow_length = min(10, length * 0.3)
        arrow_angle = 0.5  # radians

        # Calculate arrowhead points
        cos_a = np.cos(arrow_angle)
        sin_a = np.sin(arrow_angle)

        # Left arrowhead point
        left_x = end_pos[0] - arrow_length * (dx * cos_a - dy * sin_a)
        left_y = end_pos[1] - arrow_length * (dy * cos_a + dx * sin_a)

        # Right arrowhead point
        right_x = end_pos[0] - arrow_length * (dx * cos_a + dy * sin_a)
        right_y = end_pos[1] - arrow_length * (dy * cos_a - dx * sin_a)

        # Draw arrowhead
        arrow_points = [
            end_pos,
            (int(left_x), int(left_y)),
            (int(right_x), int(right_y)),
        ]
        pygame.draw.polygon(self.screen, color, arrow_points)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None
