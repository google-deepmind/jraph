# Copyright 2020 DeepMind Technologies Limited.


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Install script for setuptools."""

from setuptools import find_namespace_packages
from setuptools import setup


def _get_version():
  with open('jraph/__init__.py') as fp:
    for line in fp:
      if line.startswith('__version__') and '=' in line:
        version = line[line.find('=')+1:].strip(' \'"\n')
        if version:
          return version
    raise ValueError('`__version__` not defined in `jraph/__init__.py`')


setup(
    name='jraph',
    version=_get_version(),
    url='https://github.com/deepmind/jraph',
    license='Apache 2.0',
    author='DeepMind',
    description=('Jraph: A library for Graph Neural Networks in Jax'),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author_email='jax_graph_nets@google.com',
    keywords='jax graph neural networks python machine learning',
    packages=find_namespace_packages(exclude=['*_test.py']),
    package_data={'jraph': ['ogb_examples/test_data/*']},
    python_requires='>=3.6',
    install_requires=[
        'jax>=0.1.55',
        'jaxlib>=0.1.37',
        'numpy>=1.18.0',
    ],
    extras_require={'examples': ['dm-haiku>=0.0.2', 'absl-py>=0.9',
                                 'frozendict>=2.0.2', 'optax>=0.0.1',
                                 'scipy>=1.2.1'],
                    'ogb_examples': ['dm-haiku>=0.0.2', 'absl-py>=0.9',
                                     'optax>=0.0.1', 'pandas>=1.0.5',
                                     'dm-tree>=0.1.5']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
