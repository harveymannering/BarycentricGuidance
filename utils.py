import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.spatial import Delaunay

DEFAULT_MESH_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "va_mesh.csv")

PROMPT_TEMPLATE = (
    "Change this persons facial expression to be {} while maintaining "
    "the same facial features, composition, lighting, and camera angle"
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_va_mesh(mesh_csv=DEFAULT_MESH_CSV):
    """Return (emotions, points, Delaunay triangulation, prompts) for the VA mesh.

    mesh_csv: CSV with columns Emotion,Pleasure_Mean,Arousal_Mean (valence, arousal).
    """
    df = pd.read_csv(mesh_csv)
    emotions = df["Emotion"].tolist()
    points = df[["Pleasure_Mean", "Arousal_Mean"]].to_numpy()
    prompts = [PROMPT_TEMPLATE.format(e) for e in emotions]
    return emotions, points, Delaunay(points), prompts


def get_triangular_interpolation(p, points, tri_mesh):
    """Find the triangle containing each point in p (N, 2) and its barycentric weights.

    Points outside the mesh are assigned to the triangle with the nearest centroid
    (the weights then extrapolate). Returns (vertex indices (N, 3), weights (N, 3)).
    """
    simplex_index = tri_mesh.find_simplex(p)

    # Points outside all triangles -> nearest triangle centre
    mask = np.array(simplex_index == -1).astype(np.float32)
    triangle_centers = np.mean(points[tri_mesh.simplices], axis=1)
    distances = np.linalg.norm(triangle_centers[:, None, :] - p[None], axis=2)
    nearest_simplex = np.argmin(distances, axis=0)
    simplex_index = ((1 - mask) * simplex_index + mask * nearest_simplex).astype(np.uint8)

    point_indices = tri_mesh.simplices[simplex_index]
    selected_points = points[point_indices]

    # Solve [A B C; 1 1 1] w = [v a 1] for each point (Eq. 1 in the paper)
    barycentric_coords = np.zeros((selected_points.shape[0], 3))
    for s, (A, B, C) in enumerate(selected_points):
        M = np.array([
            [A[0], B[0], C[0]],
            [A[1], B[1], C[1]],
            [1,    1,    1   ],
        ])
        barycentric_coords[s] = np.linalg.solve(M, np.array([p[s, 0], p[s, 1], 1]))

    return point_indices, barycentric_coords


def polar_to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta)
