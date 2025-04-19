from learning.environments.salp.utils import (
    generate_target_points,
    rotate_points,
)
import torch
import math
import matplotlib.pyplot as plt

points = generate_target_points(0, 0, 10, [-45, 45], 1.0)

angle = math.radians(30)  # 30° counter‑clockwise
rot_xy = rotate_points(points, angle)

# --- plot ---------------------------------------------------------
plt.scatter(rot_xy[:, 0].cpu(), rot_xy[:, 1].cpu(), marker="o")  # x‑coords  # y‑coords
plt.xlabel("x")
plt.ylabel("y")
plt.title("Point cloud")
plt.gca().set_aspect("equal")  # optional: square axes
plt.show()

print(points)
