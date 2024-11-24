from setuptools import setup, find_packages

setup(
  name='skrrydg_ab_ml_challenge',
  version='0.0.1',
  packages=find_packages(where="src"),
  package_dir={"": "src"},
)