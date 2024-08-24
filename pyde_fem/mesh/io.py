import numpy as np


def _load_data(file_name: str, section_start: str, section_end: str, dtype=float) -> np.ndarray:
    try:
        with open(file_name) as f:
            lines = f.readlines()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File {file_name} not found.") from e

    try:
        start_index = lines.index(section_start + "\n") + 1
        end_index = lines.index(section_end + "\n", start_index)
    except ValueError as e:
        raise ValueError(f"Section {section_start} or {section_end} not found in {file_name}.") from e

    section = lines[start_index + 1 : end_index]
    return np.array([line.split()[1:] for line in section], dtype=dtype)


def _load_vtx(mesh_file: str) -> np.ndarray:
    return _load_data(mesh_file, "$Nodes", "$EndNodes")


def _load_elt(mesh_file: str) -> np.ndarray:
    return _load_data(mesh_file, "$Elements", "$EndElements", dtype=int)


def load(mesh_file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads both vertex and element data from a mesh file.

    Parameters
    ----------
    mesh_file : str
        Name of the mesh file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing arrays of vertex coordinates and element indices.

    Raises
    ------
    FileNotFoundError
        If the specified file is not found.

    Example
    -------
    mesh.msh:
    $Nodes
    4
    0 0.0 0.0
    1 1.0 0.0
    2 1.0 1.0
    3 0.0 1.0
    $EndNodes
    $Elements
    2
    0 0 1 2
    1 0 2 3
    $EndElements

    >>> import pyde_fem as pf
    >>> vertices, indices = pf.mesh.load("mesh.msh")
    >>> vertices
    array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    >>> indices
    array([[0, 1, 2], [0, 2, 3]])
    """
    vertices = _load_vtx(mesh_file)
    indices = _load_elt(mesh_file)
    return vertices, indices


def save(mesh_file: str, vertices: np.ndarray, indices: np.ndarray) -> None:
    """
    Saves mesh data to a file.

    Parameters
    ----------
    mesh_file : str
        Name of the mesh file to save.
    vertices : np.ndarray
        Array containing coordinates of mesh vertices.
    indices : np.ndarray
        Array containing indices of mesh elements.

    Raises
    ------
    FileNotFoundError
        If the specified file cannot be written to.

    Example
    -------
    >>> import pyde_fem as pf
    >>> vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    >>> indices = np.array([[0, 1, 2], [0, 2, 3]])
    >>> pf.mesh.save("mesh.msh", vertices, indices)
    """
    try:
        with open(mesh_file, "w") as f:
            f.write("$Nodes\n")
            f.write(f"{vertices.size}\n")
            for i, (x, y) in enumerate(vertices):
                f.write(f"{i} {x} {y}\n")
            f.write("$EndNodes\n")

            f.write("$Elements\n")
            f.write(f"{indices.size}\n")
            for i, elt in enumerate(indices):
                f.write(f'{i} {" ".join(map(str, elt))}\n')
            f.write("$EndElements\n")
    except IOError as e:
        raise IOError(f"Could not write to file {mesh_file}.") from e
