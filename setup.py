from setuptools import setup, find_packages


setup(
    name='idpet',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "numpy>=1.15.4",
        "pandas",
        "mdtraj",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "umap-learn",
        "requests",
        "numba",
    ],
    author= "Hamidreza Ghafouri, Giacomo Janson, Nikola Ivanovic, Alexander Miguel Monzon", 
    description="A python package for studying conformational ensembles of disordered proteins",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown"

)