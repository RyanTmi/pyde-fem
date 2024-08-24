import numpy as np
from scipy.sparse import coo_matrix


def _measure(vertices: np.ndarray) -> np.ndarray:
    """
    Calculate the measure (area or length) of the element defined by the vertices.

    Parameters
    ----------
    vertices : np.ndarray
        The vertices of the element. Shape must be (n, 2) for 2D or (n, 3) for 3D.

    Returns
    -------
    np.ndarray
        The measure (area for 3D, length for 2D) of the element.
    """
    if len(vertices.shape) == 2:
        vertices = vertices[np.newaxis, :]

    d = vertices.shape[1]
    if d == 3:
        m = np.cross(vertices[:, 0] - vertices[:, 1], vertices[:, 0] - vertices[:, 2]) / 2.0
    elif d == 2:
        m = np.linalg.norm(vertices[:, 0] - vertices[:, 1])
    else:
        raise ValueError("vertices must be 2 or 3 dimensional")

    return m.ravel()


def mass_local(vertices: np.ndarray, element: np.ndarray) -> np.ndarray:
    """
    Compute the local mass matrix for an element.

    Parameters
    ----------
    vertices : np.ndarray
        The array of vertices (nodes) coordinates.
    element : np.ndarray
        The array of vertex indices defining the element.

    Returns
    -------
    np.ndarray
        The local mass matrix for the element.

    Example
    -------
    >>> import pyde_fem as pf
    >>> vertices = np.array([[0, 0], [1, 0], [0, 1]])
    >>> element = np.array([0, 1, 2])
    >>> pf.mass_local(vertices, element)
    array([[0.08333333, 0.04166667, 0.04166667],
           [0.04166667, 0.08333333, 0.04166667],
           [0.04166667, 0.04166667, 0.08333333]])
    """
    n = element.size
    m = np.ones((n, n)) + np.eye(n, n)
    return m * _measure(vertices[element]) / (6 * (n - 1))


def stiffness_local(vtx: np.ndarray, e: np.ndarray) -> np.ndarray:
    """
    Compute the local stiffness matrix for an element.

    Parameters
    ----------
    vtx : np.ndarray
        The array of vertices (nodes) coordinates.
    e : np.ndarray
        The array of vertex indices defining the element.

    Returns
    -------
    np.ndarray
        The local stiffness matrix for the element.

    Example
    -------
    >>> import pyde_fem as pf
    >>> vertices = np.array([[0, 0], [1, 0], [0, 1]])
    >>> element = np.array([0, 1, 2])
    >>> pf.stiffness_local(vertices, element)
    array([[  1., -0.5, -0.5],
           [-0.5,  0.5,  0. ],
           [-0.5,  0. ,  0.5]])
    """
    edge_x = np.array(vtx[e[[1, 2, 0]]])
    edge_y = np.array(vtx[e[[2, 0, 1]]])

    n = np.cross([0, 0, 1], np.column_stack([edge_x - edge_y, [0, 0, 0]]))[:, :-1]
    ff = n / np.sum((vtx[e] - edge_x) * n, axis=1)[:, np.newaxis]

    return _measure(vtx[e]) * np.dot(ff, ff.T)


def assemble_global(vertices: np.ndarray, indices: np.ndarray, local) -> coo_matrix:
    """
    Assemble the global matrix from the local element matrices.

    Parameters
    ----------
    vertices : np.ndarray
        The array of vertices (nodes) coordinates.
    indices : np.ndarray
        The array of elements, each element is a list of vertex indices.
    local : function
        The function to compute the local matrix (mass or stiffness).

    Returns
    -------
    coo_matrix
        The assembled global sparse matrix.
    """
    row = np.zeros(indices.shape[1] ** 2 * indices.shape[0])
    col = np.zeros(indices.shape[1] ** 2 * indices.shape[0])
    data = np.zeros(indices.shape[1] ** 2 * indices.shape[0])

    l = 0
    for e in indices:
        m = local(vertices, e)
        for j in range(indices.shape[1]):
            for k in range(indices.shape[1]):
                row[l] = e[k]
                col[l] = e[j]
                data[l] = m[j, k]
                l += 1

    n = vertices.shape[0]
    return coo_matrix((data, (row, col)), shape=(n, n))


def mass(vertices: np.ndarray, indices: np.ndarray) -> coo_matrix:
    """
    Compute the global mass matrix for a given mesh.

    Parameters
    ----------
    vertices : np.ndarray
        The array of vertices (nodes) coordinates.
    indices : np.ndarray
        The array of elements, each element is a list of vertex indices.

    Returns
    -------
    coo_matrix
        The global mass matrix in sparse COO format.
    """
    return assemble_global(vertices, indices, mass_local)


def mass_opt(vertices: np.ndarray, indices: np.ndarray) -> coo_matrix:
    """
    Optimized computation of the global mass matrix for a given mesh.

    Parameters
    ----------
    vertices : np.ndarray
        The array of vertices (nodes) coordinates.
    indices : np.ndarray
        The array of elements, each element is a list of vertex indices.

    Returns
    -------
    coo_matrix
        The global mass matrix in sparse COO format.
    """
    def vec_elem(x: int, y: int, vs: np.ndarray) -> np.ndarray:
        return (1 + (x == y)) * vs / 12.0

    nv = vertices.shape[0]
    ne = indices.shape[0]
    d = indices.shape[1]

    k = np.zeros((d * d, ne))
    i = np.zeros((d * d, ne))
    j = np.zeros((d * d, ne))

    l = 0
    sv = _measure(vertices[indices])
    for a in range(d):
        for b in range(d):
            k[l, :] = vec_elem(a, b, sv)
            i[l, :] = indices[:, a]
            j[l, :] = indices[:, b]
            l += 1

    return coo_matrix((k.ravel(), (i.ravel(), j.ravel())), shape=(nv, nv))


def stiffness(vertices: np.ndarray, indices: np.ndarray) -> coo_matrix:
    """
    Compute the global stiffness matrix for a given mesh.

    Parameters
    ----------
    vertices : np.ndarray
        The array of vertices (nodes) coordinates.
    indices : np.ndarray
        The array of elements, each element is a list of vertex indices.

    Returns
    -------
    coo_matrix
        The global stiffness matrix in sparse COO format.
    """
    return assemble_global(vertices, indices, stiffness_local)
