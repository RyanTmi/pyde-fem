import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .io import write_mesh


def generate_mesh(
    file_name: str,
    h_sub_div: int,
    v_sub_div: int,
    h_len: float,
    v_len: float
) -> tuple[np.ndarray, np.ndarray]:
    x_vtx = np.linspace(0, h_len, h_sub_div + 1)
    y_vtx = np.linspace(0, v_len, v_sub_div + 1)

    xx, yy = np.meshgrid(x_vtx, y_vtx)
    vertices = np.array([xx.ravel(), yy.ravel()]).T

    indices = np.zeros((2 * h_sub_div * v_sub_div, 3), dtype=np.int32)
    i = 0
    for u in range(v_sub_div):
        y = (1 + h_sub_div) * u
        for x in range(y, h_sub_div + y):
            i0, i1, i2, i3 = x, x + 1, x + h_sub_div + 2, x + h_sub_div + 1
            indices[i:i + 2] = [[i0, i1, i2], [i0, i2, i3]]
            i += 2

    write_mesh(file_name, vertices, indices)
    return vertices, indices


def boundary(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boundary_faces = {}
    for i, (x0, x1, x2) in enumerate(indices):
        for j, (x, y) in enumerate([(x0, x1), (x1, x2), (x2, x0)]):
            if (y, x) in boundary_faces:
                del boundary_faces[(y, x)]
            else:
                boundary_faces[(x, y)] = 3 * i + (2 - j)

    return np.array(list(boundary_faces.keys())), np.array(list(boundary_faces.values()))


def c_component(indices: np.ndarray) -> np.ndarray:
    n = np.max(indices) + 1
    graph = np.zeros((n, n), dtype=np.int8)
    for x0, x1, x2 in indices:
        for x, y in [[x0, x1], [x1, x2], [x2, x0]]:
            graph[x, y] = 1

    graph = csr_matrix(graph)
    _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    return labels[indices[:, 0]]


def refine(vertices: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    refined_vertices: np.ndarray = vertices.copy()
    refined_indices: np.ndarray = np.zeros((4 * indices.shape[0], indices.shape[1]), dtype=int)

    sub_vertices: dict = {}
    for i, (i0, i1, i2) in enumerate(indices):
        x0, x1, x2 = vertices[[i0, i1, i2]]

        new_indices: list[int] = [0, 0, 0]
        new_vertices: list[float] = [(x1 + x2) / 2, (x0 + x2) / 2, (x0 + x1) / 2]
        for j, (x, y) in enumerate(new_vertices):
            if (x, y) in sub_vertices:
                new_indices[j] = sub_vertices[(x, y)]
            else:
                new_indices[j] = len(refined_vertices)
                sub_vertices[(x, y)] = new_indices[j]
                refined_vertices = np.append(refined_vertices, np.array([[x, y]]), axis=0)

        refined_indices[i * 4:(i + 1) * 4] = [
            [i0, new_indices[2], new_indices[1]],
            [new_indices[2], new_indices[0], new_indices[1]],
            [i1, new_indices[0], new_indices[2]],
            [new_indices[1], new_indices[0], i2]
        ]

    return refined_vertices, refined_indices
