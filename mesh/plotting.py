import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import numpy as np


def plot(
    vertices: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray = None,
    b_indices: np.ndarray = None
) -> None:
    triangulation = mtri.Triangulation(vertices[:, 0], vertices[:, 1], indices)
    if values is not None:
        plt.tricontourf(triangulation, values)
        plt.colorbar()
    elif b_indices is not None:
        start_points = vertices[b_indices[:, 0]]
        end_points = vertices[b_indices[:, 1]]
        plt.plot([start_points[:, 0], end_points[:, 0]],
                 [start_points[:, 1], end_points[:, 1]], color='red', linewidth=2)
    else:
        plt.triplot(triangulation)

    plt.show()
