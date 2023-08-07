"""setup.py for aelim."""
from setuptools import find_packages
from setuptools import setup

setup(
    name='latgmm',
    version='0.1',
    description=('Package for clustering ENSO variables in a low dimensional latent space.'),
    author='Jakob Schlör',
    author_email='jakob.schloer@uni-tuebingen.de',
    packages=find_packages()
)
