# Pyde FEM

**pyde_fem** is a Python package designed for efficient mesh manipulation and analysis, tailored for
Finite Element Method (FEM) applications and solving Partial Differential Equations (PDEs) using the
Galerkin method. With `pyde_fem`, you can easily create, modify, visualize, and analyze meshes,
making it a powerful tool for numerical simulations.

## Features

- **Mesh Creation:** Generate rectangular meshes with customizable subdivisions and dimensions.
- **Mesh Manipulation:** Refine meshes by adding midpoints to edges, and extract boundary
  information.
- **Mesh Analysis:** Calculate mass and stiffness matrices for FEM, and identify connected
  components.
- **Visualization:** Plot meshes with optional visualizations of connected components, boundary
  segments, and boundary normals.

## Installation

You can install `pyde_fem` using from PyPI pip:

```shell
pip install pyde-fem
```

## Quick Start

Here’s a quick example to get you started:

```python
import pyde_fem as pf

# Load a mesh from a file
vertices, indices = pf.mesh.load("mesh.msh")

# Plot the mesh
pf.mesh.plot(vertices, indices)
```

## Examples Usage

```python
import pyde_fem as pf
import matplotlib.pyplot as plt

# 1. Generating a Mesh:
vertices, indices = pf.mesh.generate(10, 10, 1.0, 1.0)
pf.mesh.plot(vertices, indices)
plt.show()

# 2. Generating a Mesh:
refined_vertices, refined_indices = pf.mesh.refine(vertices, indices)
pf.mesh.plot(vertices, indices)
plt.show()

# 3. Calculating Mass and Stiffness Matrices:
mass_matrix = pf.mass(vertices, indices)
stiffness_matrix = pf.stiffness(vertices, indices)


# 4. Plotting the Mesh with Boundary Information:
boundary_faces, boundary_indices = pf.mesh.boundary(indices)
boundary_normals = pf.mesh.boundary_normals(vertices, boundary_indices)

pf.mesh.plot(
    vertices,
    indices,
    boundary_indices=boundary_indices,
    boundary_normals=boundary_normals
)
plt.show()
```

## License

`pyde_fem` is licensed under the Apache Software License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any inquiries, please contact me at `timeusryan@gmail.com`.