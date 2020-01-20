from setuptools import setup, find_packages

requirements = [
    "torch>=1.2.0",
    "torchvision",
    "matplotlib",
    "numpy",
    "pre-commit",
    "black"
]

setup(
    name="Black Sheep",
    version="0.1.0",
    description="Spiking neural network framework",
    author="Erik Lanning",
    author_email="cs@eriklanning.com",
    packages=find_packages(exclude=("tests", "examples")),
    install_requires=requirements,
    python_requires=">=3.6.0",
)
