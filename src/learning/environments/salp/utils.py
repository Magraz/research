from vmas.simulator.utils import Color
import random
import numpy as np
import torch

COLOR_MAP = {
    "GREEN": Color.GREEN,
    "RED": Color.RED,
    "BLUE": Color.BLUE,
    "BLACK": Color.BLACK,
    "LIGHT_GREEN": Color.LIGHT_GREEN,
    "GRAY": Color.GRAY,
    "WHITE": Color.WHITE,
    "PURPLE": (0.75, 0.25, 0.75),
    "ORANGE": (0.75, 0.75, 0.25),
    "MAGENTA": (0.9, 0.25, 0.5),
}

COLOR_LIST = [
    Color.GREEN,
    Color.RED,
    Color.BLUE,
    Color.BLACK,
    Color.LIGHT_GREEN,
    Color.GRAY,
    Color.WHITE,
    (0.75, 0.25, 0.75),
    (0.75, 0.75, 0.25),
    (0.9, 0.25, 0.5),
]


def sample_filtered_normal(mean, std_dev, threshold):
    while True:
        # Sample a single value from the normal distribution
        value = random.normalvariate(mu=mean, sigma=std_dev)
        # Check if the value is outside the threshold range
        if abs(value) > threshold:
            return value


def generate_target_points(x, y, n_points, theta_range, d_max):
    """
    Generate n_points points starting from (x, y), where each point is positioned
    at a fixed distance (d_max) from the previous point at a random angle within theta_range.

    Parameters:
        x (float): Starting x-coordinate.
        y (float): Starting y-coordinate.
        n_points (int): Total number of points to generate.
        theta_range (tuple): Angle range in degrees (min_angle, max_angle).
        d_max (float): Fixed distance between consecutive points.

    Returns:
        list: List of tuples containing the generated (x, y) coordinates.
    """
    points = [(x, y)]  # Initialize with the starting point

    for _ in range(n_points - 1):
        # Generate a random angle within the theta_range
        theta = np.radians(np.random.uniform(theta_range[0], theta_range[1]))

        # Calculate the new point
        x_new = points[-1][0] + d_max * np.cos(theta)
        y_new = points[-1][1] + d_max * np.sin(theta)

        # Append the new point to the list
        points.append((x_new, y_new))

    return points


def batch_discrete_frechet_distance(batch_P, batch_Q):
    """
    Compute the discrete Fréchet distance between two batched tensors of points.

    Parameters:
        batch_P (torch.Tensor): Tensor of shape [B, N, 2] representing B sets of N points (x, y).
        batch_Q (torch.Tensor): Tensor of shape [B, M, 2] representing B sets of M points (x, y).

    Returns:
        torch.Tensor: Tensor of shape [B] containing the discrete Fréchet distance for each batch.
    """
    B, N, _ = batch_P.shape
    _, M, _ = batch_Q.shape

    # Initialize a large distance matrix for each batch
    ca = torch.full((B, N, M), -1.0, device=batch_P.device)

    def recursive_frechet(ca, P, Q, i, j, b):
        if ca[b, i, j] > -1:  # Use cached value
            return ca[b, i, j]

        # Compute Euclidean distance between P[i] and Q[j] for the current batch
        dist = torch.norm(P[b, i] - Q[b, j], p=2)

        if i == 0 and j == 0:  # Base case
            ca[b, i, j] = dist
        elif i == 0:  # First row
            ca[b, i, j] = torch.max(recursive_frechet(ca, P, Q, i, j - 1, b), dist)
        elif j == 0:  # First column
            ca[b, i, j] = torch.max(recursive_frechet(ca, P, Q, i - 1, j, b), dist)
        else:  # General case
            ca[b, i, j] = torch.max(
                torch.min(
                    torch.stack(
                        [
                            recursive_frechet(ca, P, Q, i - 1, j, b),
                            recursive_frechet(ca, P, Q, i - 1, j - 1, b),
                            recursive_frechet(ca, P, Q, i, j - 1, b),
                        ]
                    ),
                ),
                dist,
            )
        return ca[b, i, j]

    # Iterate over each batch and compute the Fréchet distance
    for b in range(B):
        recursive_frechet(ca, batch_P, batch_Q, N - 1, M - 1, b)

    return ca[:, -1, -1]  # Return the Fréchet distance for each batch


def angle_between_vectors(v1, v2):
    """
    Calculate the angle (in radians) between two vectors using PyTorch.

    Parameters:
        v1 (torch.Tensor): Tensor of shape [N, D] representing N vectors.
        v2 (torch.Tensor): Tensor of shape [N, D] representing N vectors.

    Returns:
        torch.Tensor: Tensor of shape [N] containing angles in radians.
    """
    # Compute dot product
    dot_product = torch.sum(v1 * v2, dim=1)

    # Compute magnitudes (L2 norms)
    norm_v1 = torch.norm(v1, p=2, dim=1)
    norm_v2 = torch.norm(v2, p=2, dim=1)

    # Compute cosine similarity
    cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-8)  # Avoid division by zero

    # Clamp values to avoid numerical errors in arccos
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Compute angle in radians
    angle = torch.acos(cos_theta)

    return angle


def is_within_any_range(number, ranges):
    """
    Check if a number is within any of the given ranges.

    Parameters:
        number (float or torch.Tensor): The number to check.
        ranges (list of tuples): List of (min, max) tuples representing the ranges.

    Returns:
        torch.Tensor (bool): True if the number is within any range, False otherwise.
    """
    # Convert ranges to a tensor of shape [N, 2]
    range_tensor = torch.tensor(ranges)  # Shape: [N, 2]

    # Extract min and max values
    range_min = range_tensor[:, 0]  # First column (min values)
    range_max = range_tensor[:, 1]  # Second column (max values)

    # Check if the number is inside any range
    inside_any_range = (number >= range_min) & (number <= range_max)

    # Return True if the number is in any range
    return torch.any(inside_any_range)


def closest_number(target, numbers):
    """
    Given a target number and a set of numbers, return the number closest to the target.

    Parameters:
        target (float or torch.Tensor): The target number.
        numbers (list of float or torch.Tensor): List of numbers to compare against.

    Returns:
        torch.Tensor: The closest number.
    """
    # Convert numbers to a PyTorch tensor
    numbers_tensor = torch.tensor(numbers)  # Shape: [N]

    # Compute absolute differences
    differences = torch.abs(numbers_tensor - target)  # Shape: [N]

    # Get the index of the minimum difference
    closest_index = torch.argmin(differences)

    # Retrieve the closest number
    closest_value = numbers_tensor[closest_index]

    return closest_value
