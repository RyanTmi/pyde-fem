import numpy as np
from scipy.sparse import coo_matrix


def volume(vtx: np.ndarray) -> float:
    if len(vtx) == 2:
        return np.linalg.norm(vtx[0] - vtx[1])
    elif len(vtx) == 3:
        return np.linalg.det(np.array([[vtx[0] - vtx[1], vtx[0] - vtx[2]]])) / 2
    else:
        raise ValueError("ValueError: vtx must be 2 or 3 dimensional")


def mass_local(vtx: np.ndarray, e: np.ndarray) -> np.ndarray:
    n = e.shape[0]
    m = np.ones((n, n)) + np.eye(n, n)
    return m * volume(vtx[e]) / (6 * (n - 1))


def mass(vtx: np.ndarray, elt: np.ndarray) -> coo_matrix:
    row = np.zeros(elt.shape[1] ** 2 * elt.shape[0])
    col = np.zeros(elt.shape[1] ** 2 * elt.shape[0])
    data = np.zeros(elt.shape[1] ** 2 * elt.shape[0])

    l = 0
    for e in elt:
        m = mass_local(vtx, e)
        for j in range(elt.shape[1]):
            for k in range(elt.shape[1]):
                row[l] = e[k]
                col[l] = e[j]
                data[l] = m[j, k]
                l += 1

    n = vtx.shape[0] if elt.shape[1] == 3 else vtx[elt].shape[0]
    return coo_matrix((data, (row, col)), shape=(n, n))


def stiffness_local(vtx: np.ndarray, e: np.ndarray) -> np.ndarray:
    d = e.shape[0]
    n = np.array(
        [
            np.cross([0, 0, 1], np.hstack([vtx[e[(i + 1) % d]] - vtx[e[(i + 2) % d]], 0]))
            for i in range(d)
        ]
    )
    n = n[:, :-1]

    ff = np.array([n[i] / np.dot(vtx[e[i]] - vtx[e[(i + 1) % d]], n[i]) for i in range(d)])
    return volume(vtx[e]) * np.dot(ff, ff.T)


def stiffness(vtx: np.ndarray, elt: np.ndarray) -> coo_matrix:
    row = np.zeros(elt.shape[1] ** 2 * elt.shape[0])
    col = np.zeros(elt.shape[1] ** 2 * elt.shape[0])
    data = np.zeros(elt.shape[1] ** 2 * elt.shape[0])

    l = 0
    for e in elt:
        m = stiffness_local(vtx, e)
        for j in range(elt.shape[1]):
            for k in range(elt.shape[1]):
                row[l] = e[k]
                col[l] = e[j]
                data[l] = m[j, k]
                l += 1

    n = vtx.shape[0] if elt.shape[1] == 3 else vtx[elt].shape[0]
    return coo_matrix((data, (row, col)), shape=(n, n))
