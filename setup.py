"""
setup.py – makes the project pip-installable.
Install with: pip install -e .
"""
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [
        line.strip() for line in f
        if line and line.strip() and not line.strip().startswith("#")
    ]

setup(
    name="telecom_analytics",
    version="1.0.0",
    description="Production-ready telecom XDR analytics platform",
    author="Telecom Analytics Team",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "telecom-pipeline=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
