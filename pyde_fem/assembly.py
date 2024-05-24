import numpy as np
from scipy.sparse import coo_matrix


def mesure(vertices: np.ndarray) -> np.ndarray:
    if len(vertices.shape) == 2:
        vertices = vertices[np.newaxis, :]

    if vertices.shape[1] == 3:
        m = np.cross(vertices[:, 0] - vertices[:, 1], vertices[:, 0] - vertices[:, 2]) / 2.0
    elif vertices.shape[1] == 2:
        m = np.linalg.norm(vertices[:, 0] - vertices[:, 1])
    else:
        raise ValueError("vertices must be 2 or 3 dimensional")
    return m.ravel()


def mass_local(vertices: np.ndarray, element: np.ndarray) -> np.ndarray:
    n = element.shape[0]
    m = np.ones((n, n)) + np.eye(n, n)
    return m * mesure(vertices[element]) / (6 * (n - 1))


def stiffness_local(vtx: np.ndarray, e: np.ndarray) -> np.ndarray:
    edge_x = np.array(vtx[e[[1, 2, 0]]])
    edge_y = np.array(vtx[e[[2, 0, 1]]])

    n = np.cross([0, 0, 1], np.column_stack([edge_x - edge_y, [0, 0, 0]]))[:, :-1]
    ff = n / np.sum((vtx[e] - edge_x) * n, axis=1)[:, np.newaxis]

    return mesure(vtx[e]) * np.dot(ff, ff.T)


def assemble_global(vertices: np.ndarray, indices: np.ndarray, local) -> coo_matrix:
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
    return assemble_global(vertices, indices, mass_local)


def mass_optv(vertices: np.ndarray, indices: np.ndarray) -> coo_matrix:
    def vec_elem(x: int, y: int, vs: np.ndarray) -> np.ndarray:
        return (1 + (x == y)) * vs / 12.0

    nv = vertices.shape[0]
    ne = indices.shape[0]
    d = indices.shape[1]

    k = np.zeros((d * d, ne))
    i = np.zeros((d * d, ne))
    j = np.zeros((d * d, ne))

    l = 0
    sv = mesure(vertices[indices])
    for a in range(d):
        for b in range(d):
            k[l, :] = vec_elem(a, b, sv)
            i[l, :] = indices[:, a]
            j[l, :] = indices[:, b]
            l += 1

    return coo_matrix((k.ravel(), (i.ravel(), j.ravel())), shape=(nv, nv))


def stiffness(vertices: np.ndarray, indices: np.ndarray) -> coo_matrix:
    return assemble_global(vertices, indices, stiffness_local)
