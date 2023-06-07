"""Install script for setuptools."""

import os
from setuptools import find_namespace_packages
from setuptools import setup

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_version():
    with open(os.path.join(_CURRENT_DIR, "army_knife", "__init__.py")) as fp:
        for line in fp:
            if line.startswith("__version__") and "=" in line:
                version = line[line.find("=") + 1 :].strip(" '\"\n")
                if version:
                    return version
        raise ValueError("`__version__` not defined in `army_knife/__init__.py`")


def _parse_requirements(path):
    with open(os.path.join(_CURRENT_DIR, path)) as f:
        return [line.rstrip() for line in f if not (line.isspace() or line.startswith("#"))]


setup(
    name="army-knife",
    version=_get_version(),
    url="https://github.com/evangelos-ch/army-knife",
    license="Apache 2.0",
    author="Evangelos Chatzaroulas",
    description=("Army Knife: Utilities for running ML experiments with JAX."),
    long_description=open(os.path.join(_CURRENT_DIR, "README.md")).read(),
    long_description_content_type="text/markdown",
    author_email="me@evangelos.ai",
    keywords="python machine learning experiments logging jax",
    packages=find_namespace_packages(exclude=["*_test.py"]),
    install_requires=_parse_requirements(os.path.join(_CURRENT_DIR, "requirements", "requirements.txt")),
    tests_require=_parse_requirements(os.path.join(_CURRENT_DIR, "requirements", "requirements-tests.txt")),
    zip_safe=False,
    include_package_data=True,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
