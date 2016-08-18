from setuptools import setup
from distutils.core import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy as np
import os


requirements = ['setuptools>=18.0', 'cython', 'numpy', 'scipy', 'pandas',
                'matplotlib', 'seaborn', 'pytest', 'freeze']

SRC = os.path.join('cxx', 'src')
INC = os.path.join('cxx', 'include')

extensions = [
    Extension('baxcat.state',
              sources=[os.path.join('baxcat', 'interface', 'state.pyx'),
                       os.path.join(SRC, 'state.cpp'),
                       os.path.join(SRC, 'view.cpp'),
                       os.path.join(SRC, 'categorical.cpp'),
                       os.path.join(SRC, 'continuous.cpp'),
                       os.path.join(SRC, 'feature_tree.cpp')],
              extra_compile_args=['-std=c++11', '-Wno-comment', '-fopenmp'],
              extra_link_args=['-lstdc++'],
              include_dirs=[SRC, INC, np.get_include()],
              language="c++"),
    Extension('baxcat.dist.nng',
              sources=[os.path.join('baxcat', 'dist', 'nng.pyx')],
              extra_compile_args=['-std=c++11', '-fopenmp'],
              extra_link_args=['-lstdc++'],
              include_dirs=[INC, np.get_include()],
              language="c++"),
    Extension('baxcat.dist.csd',
              sources=[os.path.join('baxcat', 'dist', 'csd.pyx')],
              extra_compile_args=['-std=c++11', '-fopenmp'],
              extra_link_args=['-lstdc++'],
              include_dirs=[INC, np.get_include()],
              language="c++")]

setup(
    name='baxcat',
    version='0.0.1',
    author='Baxter S. Eaves Jr.',
    url='TBA',
    long_description='TBA.',
    package_dir={'baxcat': 'baxcat/'},
    setup_requires=requirements,
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': build_ext}
)
