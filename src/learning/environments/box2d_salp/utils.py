import numpy as np

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


def get_linear_positions(n_agents):
    positions = []
    for i in range(n_agents):
        positions.append((10 + i * 1.5, 10))

    return positions


def get_scatter_positions(world_width, world_height, n_agents, min_distance=1.0):
    """
    Generate random starting positions for all agents

    Args:
        min_distance (float): Minimum distance between agents when maintain_chain=False

    Returns:
        list: List of (x, y) tuples for each agent position
    """
    positions = []

    # Define safe boundaries (away from walls)
    margin = 2.0  # Distance from walls
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
