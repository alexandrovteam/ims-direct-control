from setuptools import setup, find_packages

from remote_control import __version__

setup(name='remote_control',
      version=__version__,
      description='Python library for controlling the TransMIT APS-MALDI source remotely',
      url='',
      author='Andrew Palmer',
      packages=find_packages(),
      install_requires=[
          'pandas',
          'numpy',
          'scipy',
          'pexpect',
          'pyMSpec',
          'matplotlib', #==2.0.2',
          'scikit-image',
          'scikit-learn'
      ])
