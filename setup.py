""" Quantum Inspire examples

Copyright 2018-2021 QuTech Delft

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
from setuptools import setup


def get_version_number():
    """ Extract the version number from the source code.

        Returns:
            str: the version number.
    """
    with open('version.py', 'r') as file_stream:
        line = file_stream.readline().split()
        version_number = line[2].replace('\'', '')
    return version_number


def get_long_description():
    """ Extract the long description from the README file """

    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()

    return long_description


setup(name='quantum-inspire-examples',
      description='Quantum Inspire platform examples',
      long_description=get_long_description(),
      long_description_content_type='text/markdown',
      version=get_version_number(),
      author='QuantumInspire',
      python_requires='>=3.7',
      package_dir={},
      data_files=[],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'License :: OSI Approved :: Apache Software License'],
      license='Apache 2.0',
      packages=[''],
      install_requires=['boolean.py', 'jupyter', 'matplotlib>=2.1', 'numpy>=1.17', 'projectq>=0.4', 'qiskit>=0.20.0',
                        'nbimporter', 'quantuminspire'],
      extras_require={
          'rtd': ['sphinx', 'sphinx-rtd-theme', 'jupyter-sphinx', 'nbsphinx', 'sphinx-automodapi',
                  'sphinx-copybutton'],
      })
