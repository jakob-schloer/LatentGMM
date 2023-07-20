"""setup.py for aelim."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='stfor',
    version='0.1',
    description=('Package for Spatio-temporal forecasting of monthly data.'),
    author='Jakob Schlör',
    author_email='jakob.schloer@uni-tuebingen.de',
    packages=find_packages()
)
