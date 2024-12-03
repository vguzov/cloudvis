import os
from glob import glob
from setuptools import setup, find_packages
version = '1.6.0'

with open("README.md", "r") as fi:
    long_description = fi.read()

keywords = ["rendering", "pointcloud", "opengl", "mesh"]

classifiers = [
        'Intended Audience :: Developers',
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3'
    ]

requirements = [
    "cloudrender>=1.3.2",
    "pygame",
    "pyglm>=2.2.0",
    "imgui[glfw]>=2.0.0",
    "trimesh",
    "tqdm",
    "zipjson",
    "loguru",
    "scikit-image",
    "palettable",
    "smplx",
    "dacite",
    "toml"
]


setup(
    name="cloudvis",
    packages=find_packages(),
    include_package_data=True,
    version=version,
    description="Interactive visualizer for pointclouds, meshes, SMPL models and more",
    author="Vladimir Guzov",
    author_email="vguzov@mpi-inf.mpg.de",
    url="https://github.com/vguzov/cloudvis",
    keywords=keywords,
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    classifiers=classifiers
)