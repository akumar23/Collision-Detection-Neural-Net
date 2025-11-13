"""
Setup file for Robot Navigation Package
"""

from setuptools import setup, find_packages

setup(
    name="robot-navigation",
    version="1.0.0",
    description="Neural network-based collision detection and path planning for autonomous robots",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn",
        "pygame",
        "pymunk",
        "noise",
        "matplotlib",
    ],
)

