import numpy as np


def print_file(file_name: str) -> None:
    with open(file_name) as f:
        print(f.read())


def load_data(
    file_name: str,
    section_start: str,
    section_end: str,
    dtype=float
) -> np.ndarray:
    try:
        with open(file_name) as f:
            lines = f.readlines()
    except FileNotFoundError as e:
        print(e.strerror)

    start_index = lines.index(section_start + '\n') + 1
    end_index = lines.index(section_end + '\n', start_index)
    section = lines[start_index + 1:end_index]
    return np.array([line.split()[1:] for line in section], dtype=dtype)


def load_vtx(mesh_file: str) -> np.ndarray:
    return load_data(mesh_file, '$Noeuds', '$FinNoeuds')


def load_elt(mesh_file: str) -> np.ndarray:
    return load_data(mesh_file, '$Elements', '$FinElements', dtype=int)


def load(mesh_file: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        with open(mesh_file) as f:
            lines = f.readlines()
    except FileNotFoundError as e:
        print(e.strerror)

    start_index = lines.index('$Noeuds\n') + 1
    end_index = lines.index('$FinNoeuds\n', start_index)
    section = lines[start_index + 1:end_index]
    vertices = np.array([line.split()[1:] for line in section], dtype=float)

    start_index = lines.index('$Elements\n') + 1
    end_index = lines.index('$FinElements\n', start_index)
    section = lines[start_index + 1:end_index]
    indices = np.array([line.split()[1:] for line in section], dtype=int)
    return vertices, indices


def save(mesh_file: str, vertices: np.ndarray, indices: np.ndarray) -> None:
    try:
        with open(mesh_file, 'w') as f:
            f.write('$Noeuds\n')
            f.write(f'{len(vertices)}\n')
            for i, (x, y) in enumerate(vertices):
                f.write(f'{i} {x} {y}\n')
            f.write('$FinNoeuds\n')

            f.write('$Elements\n')
            f.write(f'{len(indices)}\n')
            for i, elt in enumerate(indices):
                f.write(f'{i} {" ".join(map(str, elt))}\n')
            f.write('$FinElements\n')
    except FileNotFoundError as e:
        print(e.strerror)
