import numpy as np
from Box2D import b2ContactListener

AGENT_CATEGORY = 0x0001  # Binary: 0001
BOUNDARY_CATEGORY = 0x0002  # Binary: 0010

COLORS_LIST = [
    # Primary colors
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    # Secondary colors
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (255, 192, 203),  # Pink
    (0, 128, 0),  # Dark Green
    (0, 0, 128),  # Navy
    (128, 128, 0),  # Olive
    # Tertiary colors
    (255, 99, 71),  # Tomato
    (70, 130, 180),  # Steel Blue
    (255, 20, 147),  # Deep Pink
    (32, 178, 170),  # Light Sea Green
    (255, 215, 0),  # Gold
    (138, 43, 226),  # Blue Violet
    # Earth tones
    (210, 180, 140),  # Tan
    (139, 69, 19),  # Saddle Brown
    (160, 82, 45),  # Sienna
    (205, 92, 92),  # Indian Red
    (222, 184, 135),  # Burlywood
    (188, 143, 143),  # Rosy Brown
    # Cool colors
    (95, 158, 160),  # Cadet Blue
    (72, 61, 139),  # Dark Slate Blue
    (123, 104, 238),  # Medium Slate Blue
    (0, 191, 255),  # Deep Sky Blue
    (30, 144, 255),  # Dodger Blue
    (100, 149, 237),  # Cornflower Blue
    # Warm colors
    (255, 69, 0),  # Red Orange
    (255, 140, 0),  # Dark Orange
]


class UnionFind:
    """Union-Find data structure for efficiently tracking connected components"""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # Already connected

        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)


class BoundaryContactListener(b2ContactListener):
    """Contact listener to detect collisions between agents and boundaries"""

    def __init__(self):
        super().__init__()
        self.boundary_collision = False

    def BeginContact(self, contact):
        # Check if this is a collision between an agent and boundary
        fixture_a, fixture_b = contact.fixtureA, contact.fixtureB

        category_a = fixture_a.filterData.categoryBits
        category_b = fixture_b.filterData.categoryBits

        # Check if one fixture is an agent and the other is a boundary
        if (category_a == AGENT_CATEGORY and category_b == BOUNDARY_CATEGORY) or (
            category_b == AGENT_CATEGORY and category_a == BOUNDARY_CATEGORY
        ):
            self.boundary_collision = True

    def reset(self):
        """Reset collision flag"""
        self.boundary_collision = False


def get_linear_positions(n_agents):
    positions = []
    for i in range(n_agents):
        positions.append((10 + i * 1.5, 10))

    return positions


def get_scatter_positions(world_width, world_height, n_agents, min_distance=0.5):
    """
    Generate random starting positions for all agents

    Args:
        min_distance (float): Minimum distance between agents when maintain_chain=False

    Returns:
        list: List of (x, y) tuples for each agent position
    """
    positions = []

    # Define safe boundaries (away from walls)
    margin = 12.0  # Distance from walls
    safe_x_min = margin
    safe_x_max = world_width - margin
    safe_y_min = margin
    safe_y_max = world_height - margin

    # Completely random positions with minimum distance constraint
    for i in range(n_agents):
        attempts = 0
        max_attempts = 100

        while attempts < max_attempts:
            pos_x = np.random.uniform(safe_x_min, safe_x_max)
            pos_y = np.random.uniform(safe_y_min, safe_y_max)

            # Check minimum distance from other agents
            valid_position = True
            for existing_pos in positions:
                distance = np.sqrt(
                    (pos_x - existing_pos[0]) ** 2 + (pos_y - existing_pos[1]) ** 2
                )
                if distance < min_distance:
                    valid_position = False
                    break

            if valid_position:
                positions.append((pos_x, pos_y))
                break

            attempts += 1

        # Fallback if we can't find a valid position
        if attempts >= max_attempts:
            pos_x = safe_x_min + (i * (safe_x_max - safe_x_min) / n_agents)
            pos_y = safe_y_min + np.random.uniform(0, safe_y_max - safe_y_min)
            positions.append((pos_x, pos_y))

    return positions


def position_target_area(
    width, height, existing_positions=None, min_distance=8.0, max_attempts=100
):
    """
    Position a target area with direction-based offset from the world center,
    ensuring minimum distance from existing targets.

    Args:
        width (float): World width
        height (float): World height
        existing_positions (list): List of (x, y) tuples of existing target positions
        min_distance (float): Minimum distance required between target positions
        max_attempts (int): Maximum number of positioning attempts before falling back

    Returns:
        tuple: (x, y) coordinates for the target area
    """
    # Initialize existing positions if None
    if existing_positions is None:
        existing_positions = []

    # Calculate world center
    center_x = width / 2
    center_y = height / 2

    # Define bounding box dimensions (60% of world size)
    box_width = width * 0.6
    box_height = height * 0.6

    # Calculate bounding box boundaries
    box_left = center_x - box_width / 2
    box_right = center_x + box_width / 2
    box_bottom = center_y - box_height / 2
    box_top = center_y + box_height / 2

    # Margin from edges
    margin = 10
    boundary_margin = 4.0

    for attempt in range(max_attempts):
        # Step 1: Generate random position within bounding box
        x = np.random.uniform(box_left + margin, box_right - margin)
        y = np.random.uniform(box_bottom + margin, box_top - margin)

        # Step 2: Calculate direction vector from center
        dir_x = x - center_x
        dir_y = y - center_y

        # Step 3: Normalize direction vector (if not zero)
        magnitude = np.sqrt(dir_x**2 + dir_y**2)
        if magnitude > 0.001:  # Avoid division by zero
            dir_x /= magnitude
            dir_y /= magnitude

        # Step 4: Apply additional offset in same direction
        offset_magnitude = 10
        x = center_x + dir_x * (magnitude + offset_magnitude)
        y = center_y + dir_y * (magnitude + offset_magnitude)

        # Step 5: Ensure the position stays within world bounds
        x = np.clip(x, boundary_margin, width - boundary_margin)
        y = np.clip(y, boundary_margin, height - boundary_margin)

        # Step 6: Check minimum distance from all existing target positions
        valid_position = True
        for pos_x, pos_y in existing_positions:
            distance = np.sqrt((x - pos_x) ** 2 + (y - pos_y) ** 2)
            if distance < min_distance:
                valid_position = False
                break

        # If position is valid (meets minimum distance), return it
        if valid_position:
            return x, y

    # Fallback: If we couldn't find a valid position after max attempts,
    # try to place it at the furthest point from all existing positions
    if existing_positions:
        # Find position furthest from all existing positions
        best_position = None
        best_min_distance = -1

        # Try a grid of positions
        grid_size = 20
        for grid_x in np.linspace(boundary_margin, width - boundary_margin, grid_size):
            for grid_y in np.linspace(
                boundary_margin, height - boundary_margin, grid_size
            ):
                min_dist_to_existing = float("inf")
                for pos_x, pos_y in existing_positions:
                    dist = np.sqrt((grid_x - pos_x) ** 2 + (grid_y - pos_y) ** 2)
                    min_dist_to_existing = min(min_dist_to_existing, dist)

                if min_dist_to_existing > best_min_distance:
                    best_min_distance = min_dist_to_existing
                    best_position = (grid_x, grid_y)

        return best_position

    # If there are no existing positions or we couldn't find a better position,
    # just return a random position within bounds
    x = np.random.uniform(boundary_margin, width - boundary_margin)
    y = np.random.uniform(boundary_margin, height - boundary_margin)
    return x, y
