# Pyde FEM

**pyde_fem** is a Python package for mesh manipulation, a submodule of the larger pyde_fem project for Finite Element Method (FEM) analysis and solving Partial Differential Equations (PDEs) using the Galerkin method.

## Features

- **Mesh Manipulation:** Easily create, modify, and visualize meshes for FEM analysis.

## Installation

You can install `pyde_fem` using pip:

```shell
pip3 install pyde_fem
```

## Usage

```python
import pyde_fem as pf

vertices, indices = pf.mesh.load('mesh.msh')
pf.mesh.plot(vertices, indices)
```
