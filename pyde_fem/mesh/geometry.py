import numpy as np

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from .io import save


def generate(
    file_name: str,
    h_sub_div: int,
    v_sub_div: int,
    h_len: float,
    v_len: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a rectangle mesh with specified subdivision parameters and dimensions.

    Parameters
    ----------
    file_name : str
        Name of the file to save the generated mesh.
    h_sub_div : int
        Number of horizontal subdivisions.
    v_sub_div : int
        Number of vertical subdivisions.
    h_len : float
        Length of the horizontal dimension.
    v_len : float
        Length of the vertical dimension.

    Returns
    -------
    vertices : np.ndarray
        Array containing coordinates of mesh vertices.
    indices : np.ndarray
        Array containing indices of mesh elements.
    """
    x_vtx = np.linspace(0, h_len, h_sub_div + 1)
    y_vtx = np.linspace(0, v_len, v_sub_div + 1)

    xx, yy = np.meshgrid(x_vtx, y_vtx)
    vertices = np.array([xx.ravel(), yy.ravel()]).T

    indices = np.zeros((2 * h_sub_div * v_sub_div, 3), dtype=np.uint32)
    i = 0
    for u in range(v_sub_div):
        y = (1 + h_sub_div) * u
        for x in range(y, h_sub_div + y):
            i0, i1, i2, i3 = x, x + 1, x + h_sub_div + 2, x + h_sub_div + 1
            indices[i:i + 2] = [[i0, i1, i2], [i0, i2, i3]]
            i += 2

    save(file_name, vertices, indices)
    return vertices, indices


def boundary(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts boundary information from mesh indices.

    Parameters
    ----------
    indices : np.ndarray
        Array containing indices of mesh elements.

    Returns
    -------
    boundary_data : tuple[np.ndarray, np.ndarray]
        Tuple containing arrays of boundary face indices and their corresponding indices in the original array.
    """
    edge_x = indices[:, [1, 2, 0]].ravel()
    edge_y = indices[:, [2, 0, 1]].ravel()

    boundary_faces = {}
    for i, (x, y) in enumerate(zip(edge_x, edge_y)):
        if (y, x) in boundary_faces:
            del boundary_faces[(y, x)]
        else:
            boundary_faces[(x, y)] = i

    return np.array(list(boundary_faces.keys())), np.array(list(boundary_faces.values()))


def c_component(indices: np.ndarray) -> np.ndarray:
    """
    Identifies connected components in a mesh.

    Parameters
    ----------
    indices : np.ndarray
        Array containing indices of mesh elements.

    Returns
    -------
    labels: np.ndarray
        Array containing labels for connected components.
    """
    n = np.max(indices) + 1

    if indices.shape[1] == 3:
        rows = indices[:, [0, 1, 2]].ravel()
        cols = indices[:, [1, 2, 0]].ravel()
    else:  # Assuming that indices.shape[1] == 2
        rows = indices[:, 0]
        cols = indices[:, 1]

    data = np.ones(len(rows), dtype=np.uint8)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    _, labels = connected_components(csgraph=graph, directed=False)
    return labels[indices[:, 0]]


def c_component_dual(indices: np.ndarray) -> np.ndarray:
    """
    Identifies connected components in the dual graph of a mesh.

    Parameters
    ----------
    indices : np.ndarray
        Array containing indices of mesh elements.

    Returns
    -------
    labels: np.ndarray
        Array containing labels for connected components.
    """
    edge_x = indices[:, [1, 2, 0]].ravel()
    edge_y = indices[:, [2, 0, 1]].ravel()
    common_edges = {}

    n = len(indices)
    graph = lil_matrix(np.zeros((n, n), dtype=np.uint8))

    for i, (x, y) in enumerate(zip(edge_x, edge_y)):
        if (y, x) in common_edges:
            graph[i // 3, common_edges[(y, x)]] = 1
            del common_edges[(y, x)]
        else:
            common_edges[(x, y)] = i // 3

    _, labels = connected_components(csgraph=graph, directed=False)
    return labels


def refine(vertices: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine a mesh by adding midpoints to edges.

    Parameters
    ----------
    vertices : np.ndarray
        Array containing coordinates of mesh vertices.
    indices : np.ndarray
        Array containing indices of mesh elements.

    Returns
    -------
    refined_vertices : np.ndarray
        Array containing coordinates of refined mesh vertices.
    refined_indices : np.ndarray
        Array containing indices of refined mesh elements.
    """
    refined_vertices = vertices.copy()
    refined_indices = np.zeros((4 * indices.shape[0], indices.shape[1]), dtype=int)

    sub_vertices = {}
    for i, t in enumerate(indices):
        x0, x1, x2 = vertices[t]

        new_indices = np.zeros(3, dtype=int)
        new_vertices = np.array([(x1 + x2) / 2, (x0 + x2) / 2, (x0 + x1) / 2])
        for j, (x, y) in enumerate(new_vertices):
            if (x, y) in sub_vertices:
                new_indices[j] = sub_vertices[(x, y)]
            else:
                new_indices[j] = len(refined_vertices)
                sub_vertices[(x, y)] = new_indices[j]
                refined_vertices = np.vstack([refined_vertices, [[x, y]]])

        refined_indices[i * 4:(i + 1) * 4] = [
            [t[0], new_indices[2], new_indices[1]],
            [new_indices[2], new_indices[0], new_indices[1]],
            [t[1], new_indices[0], new_indices[2]],
            [new_indices[1], new_indices[0], t[2]]
        ]

    return refined_vertices, refined_indices
