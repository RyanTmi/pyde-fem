import numpy as np
from scipy.sparse import csgraph, csr_matrix, lil_matrix

from .io import save


def generate(
    h_sub_div: int, v_sub_div: int, h_len: float, v_len: float, file_name: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a rectangular mesh with specified subdivisions and dimensions, and save it to a file.

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
    tuple[np.ndarray, np.ndarray]
        Tuple containing coordinates of the generated mesh and indices of mesh elements.

    Example
    -------
    >>> import pyde_fem as pf
    >>> vertices, indices = pf.mesh.generate(2, 2, 1.0, 1.0)
    >>> vertices
    array([[0. , 0. ],
           [0.5, 0. ],
           [1. , 0. ],
           [0. , 0.5],
           [0.5, 0.5],
           [1. , 0.5],
           [0. , 1. ],
           [0.5, 1. ],
           [1. , 1. ]])
    >>> indices
    array([[0, 1, 4],
           [0, 4, 3],
           [1, 2, 5],
           [1, 5, 4],
           [3, 4, 7],
           [3, 7, 6],
           [4, 5, 8],
           [4, 8, 7]], dtype=uint32)
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
            indices[i : i + 2] = [[i0, i1, i2], [i0, i2, i3]]
            i += 2

    if file_name is not None:
        save(file_name, vertices, indices)

    return vertices, indices


def boundary(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract boundary edges from mesh indices.

    Parameters
    ----------
    indices : np.ndarray
        Array containing indices of mesh elements.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing arrays of boundary edge indices and their corresponding indices in the
        original array.

    Example
    -------
    >>> import pyde_fem as pf
    >>> indices = np.array([[0, 1, 2], [2, 3, 0], [1, 4, 2]])
    >>> boundary_edges, boundary_indices = pf.mesh.boundary(indices)
    >>> boundary_edges
    array([[0, 1],
           [1, 4],
           [4, 2],
           [2, 3],
           [3, 0]])
    >>> boundary_indices
    array([0, 2, 5, 7, 8])
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


def boundary_normals(vertices: np.ndarray, boundary_indices: np.ndarray) -> np.ndarray:
    """
    Compute the normal vectors of the boundary edges.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (n, 2) containing coordinates of the mesh vertices.
    boundary_indices : np.ndarray
        Array of shape (m, 2) containing indices of boundary edges.

    Returns
    -------
    np.ndarray
        Array of shape (m, 2) containing the normal vectors of the boundary edges.

    Example
    -------
    >>> import pyde_fem as pf
    >>> vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> boundary_indices = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    >>> pf.mesh.boundary_normals(vertices, boundary_indices)
    array([[ 0., -1.],
           [ 1.,  0.],
           [ 0.,  1.],
           [-1.,  0.]])
    """
    boundary_vertices = vertices[boundary_indices]
    dx = boundary_vertices[:, 1, 0] - boundary_vertices[:, 0, 0]
    dy = boundary_vertices[:, 1, 1] - boundary_vertices[:, 0, 1]
    normals = np.column_stack((dy, -dx))
    norms = np.linalg.norm(normals, axis=1)
    return normals / norms[:, np.newaxis]


def connected_component(indices: np.ndarray) -> np.ndarray:
    """
    Identify connected components in a mesh based on shared vertices.

    Parameters
    ----------
    indices : np.ndarray
        Array containing indices of mesh elements.

    Returns
    -------
    np.ndarray
        Array containing labels for connected components.

    Example
    -------
    >>> import pyde_fem as pf
    >>> indices = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])
    >>> pf.mesh.connected_component(indices)
    array([0, 0, 1])
    """
    n = np.max(indices) + 1

    if indices.shape[1] == 3:
        rows = indices[:, [0, 1, 2]].ravel()
        cols = indices[:, [1, 2, 0]].ravel()
    elif indices.shape[1] == 2:
        rows = indices[:, 0]
        cols = indices[:, 1]
    else:
        raise ValueError("Indices elements must be 2 or 3 dimensional")

    data = np.ones(len(rows), dtype=np.uint8)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    _, labels = csgraph.connected_components(csgraph=graph, directed=False)
    return labels[indices[:, 0]]


def c_component_dual(indices: np.ndarray) -> np.ndarray:
    """
    Identify connected components in the dual graph of a mesh.

    Parameters
    ----------
    indices : np.ndarray
        Array containing indices of mesh elements.

    Returns
    -------
    np.ndarray
        Array containing labels for connected components in the dual graph.

    Example
    -------
    >>> import pyde_fem as pf
    >>> indices = np.array([[0, 1, 2], [2, 3, 0], [1, 4, 2]])
    >>> pf.mesh.c_component_dual(indices)
    array([0, 0, 1], dtype=int32)
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

    _, labels = csgraph.connected_components(csgraph=graph, directed=False)
    return labels


def refine(vertices: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine a mesh by adding midpoints to the edges of each element.

    Parameters
    ----------
    vertices : np.ndarray
        Array containing coordinates of mesh vertices.
    indices : np.ndarray
        Array containing indices of mesh elements.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing coordinates of the refined mesh and indices of refined mesh elements.

    Example
    -------
    >>> import pyde_fem as pf
    >>> vertices = np.array([[0, 0], [1, 0], [0, 1]])
    >>> indices = np.array([[0, 1, 2]])
    >>> new_vertices, new_indices = pf.mesh.refine(vertices, indices)
    >>> new_vertices
    array([[0. , 0. ],
           [1. , 0. ],
           [0. , 1. ],
           [0.5, 0. ],
           [0.5, 0.5],
           [0. , 0.5]])
    >>> new_indices
    array([[0, 3, 5],
           [3, 4, 5],
           [1, 4, 3],
           [5, 4, 2]])
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

        refined_indices[i * 4 : (i + 1) * 4] = [
            [t[0], new_indices[2], new_indices[1]],
            [new_indices[2], new_indices[0], new_indices[1]],
            [t[1], new_indices[0], new_indices[2]],
            [new_indices[1], new_indices[0], t[2]],
        ]

    return refined_vertices, refined_indices
