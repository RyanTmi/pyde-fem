from pathlib import Path
from setuptools import setup, find_packages


def main() -> None:
    setup(
        name="pyde_fem",
        version="0.5.1",
        description="Python package for Finite Element Method (FEM) for solving Partial Differential Equations (PDEs)",
        long_description=Path("README.md").read_text(),
        long_description_content_type="text/markdown",
        url="https://github.com/RyanTmi/pyde_fem",
        author="Ryan Timeus",
        author_email="timeusryan@gmail.com",
        license="Apache License 2.0",
        project_urls={
            "Source": "https://github.com/RyanTmi/pyde_fem",
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Mathematics",
        ],
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
    )


if __name__ == "__main__":
    main()
