import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

sns.set_theme()


def plot(vertices: np.ndarray, indices: np.ndarray, /, **kwargs):
    """
    Plot a mesh with optional connected components, boundary visualization, and boundary normals.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (n, 2) representing the vertices of the mesh.
    indices : np.ndarray
        Array of shape (m, 3) representing the indices of the vertices forming triangles.
    kwargs :
        connected_components : np.ndarray, optional
            Array of shape (m,) representing the connected components of each triangle.
        boundary_indices : np.ndarray, optional
            Array of shape (p, 2) representing the indices of the boundary segments.
        boundary_connected_components : np.ndarray, optional
            Array of shape (p,) representing the connected components of each boundary segment.
        boundary_normals : np.ndarray, optional
            Array of shape (p, 2) representing the normal vectors to the boundary segments.
        values : np.ndarray, optional
            Array of shape (n,) representing values associated with each vertex for plotting.
    """
    fig = plt.figure("Mesh", figsize=(8, 8))
    ax = fig.add_subplot()

    _plot_triangles(ax, fig, vertices, indices, **kwargs)
    _plot_boundary(ax, vertices, **kwargs)

    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.2)
    fig.show()


def _plot_triangles(ax, fig, vertices: np.ndarray, indices: np.ndarray, **kwargs):
    connected_components = kwargs.get("connected_components", np.zeros(indices.shape[0], dtype=int))
    cc_count = np.max(connected_components) + 1
    colors = plt.colormaps.get_cmap("twilight")(np.linspace(0.2, 1.0, cc_count))

    for k, c in zip(range(cc_count), colors):
        triangulation = mtri.Triangulation(vertices[:, 0], vertices[:, 1], indices[connected_components == k])
        values = kwargs.get("values")
        alpha = 1.0
        if values is not None:
            alpha = 0.5
            m = ax.tricontourf(triangulation, values)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(m, cax=cax)

        ax.triplot(triangulation, linewidth=0.5, color=c, alpha=alpha)


def _plot_boundary(ax, vertices: np.ndarray, **kwargs):
    boundary_indices = kwargs.get("boundary_indices")
    boundary_normals = kwargs.get("boundary_normals")
    if boundary_indices is None:
        if boundary_normals is not None:
            raise ValueError("Cannot plot boundary normals without boundary indices")
        return

    boundary_connected_components = kwargs.get(
        "boundary_connected_components", np.zeros(boundary_indices.shape[0], dtype=int)
    )
    bcc_count = np.max(boundary_connected_components) + 1
    bcolors = plt.colormaps.get_cmap("inferno")(np.linspace(0.5, 1.0, bcc_count))

    for k, c in zip(range(bcc_count), bcolors):
        start_points = vertices[boundary_indices[boundary_connected_components == k][:, 0]]
        end_points = vertices[boundary_indices[boundary_connected_components == k][:, 1]]
        ax.plot(
            [start_points[:, 0], end_points[:, 0]],
            [start_points[:, 1], end_points[:, 1]],
            linewidth=2,
            color=c,
        )
    if boundary_normals is not None:
        start_points = vertices[boundary_indices[:, 0]]
        end_points = vertices[boundary_indices[:, 1]]
        ax.quiver(
            (start_points[:, 0] + end_points[:, 0]) / 2.0,
            (start_points[:, 1] + end_points[:, 1]) / 2.0,
            boundary_normals[:, 0],
            boundary_normals[:, 1],
            color="teal",
            pivot="tail",
            scale_units="xy",
            units="xy",
            scale=15,
            minshaft=2.5,
        )
