import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="capstone_project",

    description="A short description of the project.",

    author="André Lukas Schorlemmer",

    packages=find_packages(exclude=['data', 'reports', 'output', 'notebooks']),

    long_description=read('README.md'),
)
