from setuptools import setup, find_packages

setup(
    name="shapes",
    version="0.0.1",
    description="Convex shape optimization library in Pytorch",
    author="Eloi Martinet",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "pyvista",
        "fiblat",
        "tqdm"
    ],
)
