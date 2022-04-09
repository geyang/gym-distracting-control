from os import path
from setuptools import setup, find_packages

with open(path.join(path.abspath(path.dirname(__file__)), 'VERSION'), encoding='utf-8') as f:
    version = f.read()

with open(path.join(path.abspath(path.dirname(__file__)), 'README'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='distracting_control',
      packages=find_packages(exclude="specs"),
      install_requires=["gym>=0.21.0", "dm_control", "numpy", ],
      description='distractor control suite contains variants of the DeepMind Control suite with visual distraction',
      long_description=long_description,
      author='Ge Yang<ge.ike.yang@gmail.com>',
      url='https://github.com/geyang/distracting_control',
      author_email='ge.ike.yang@gmail.com',
      version=version)
