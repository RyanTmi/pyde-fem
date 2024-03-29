import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import numpy as np


def plot(
    vertices: np.ndarray,
    indices: np.ndarray,
    indices_b: np.ndarray = None,
    cc: np.ndarray = None,
    ccb: np.ndarray = None,
    values: np.ndarray = None
) -> None:
    if cc is None:
        cc = np.zeros_like(indices)

    cc_count = np.max(ccb) + 1
    colors = plt.colormaps.get_cmap('twilight')(np.linspace(0.2, 1.0, cc_count))

    plt.figure()
    for k, c in zip(range(np.max(cc) + 1), colors):
        triangulation = mtri.Triangulation(vertices[:, 0], vertices[:, 1], indices[cc == k])
        if values is not None:
            plt.tricontourf(triangulation, values, alpha=0.5, cmap='cividis')

        plt.triplot(triangulation, lw=0.5, c=c)

    if indices_b is not None:
        if ccb is None:
            ccb = np.zeros_like(indices_b)

        ccb_count = np.max(ccb) + 1
        colors_b = plt.colormaps.get_cmap('inferno')(np.linspace(0.5, 1.0, ccb_count))
        for k, c in zip(range(ccb_count), colors_b):
            start_points = vertices[indices_b[ccb == k][:, 0]]
            end_points = vertices[indices_b[ccb == k][:, 1]]
            plt.plot([start_points[:, 0], end_points[:, 0]],
                     [start_points[:, 1], end_points[:, 1]], lw=3, c=c)

    plt.show()
