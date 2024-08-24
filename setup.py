from pathlib import Path

from setuptools import setup, find_packages


def main() -> None:
    setup(
        name="pyde-fem",
        version="1.0.0",
        author="Ryan Timeus",
        author_email="timeusryan@gmail.com",
        description="Python package for Finite Element Method (FEM) for solving Partial Differential Equations (PDEs)",
        long_description=Path("README.md").read_text(),
        long_description_content_type="text/markdown",
        url="https://github.com/RyanTmi/pyde-fem",
        license="Apache License 2.0",
        keywords=[
            "finite element method",
            "partial differential equations",
            "mesh",
            "numerical analysis",
        ],
        python_requires=">=3.8",
        install_requires=["matplotlib", "numpy>=1.26", "scipy"],
        packages=find_packages(),
        package_data={"": ["README.md"]},
        include_package_data=True,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "License :: OSI Approved :: Apache Software License",
        ],
    )


if __name__ == "__main__":
    main()
