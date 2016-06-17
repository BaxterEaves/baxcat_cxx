from setuptools import setup
from distutils.core import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy as np

cmdclass = dict()
cmdclass['build_ext'] = build_ext
boost_dir = '/usr/local/Cellar/boost/1.56.0'

extensions = [Extension('baxcat.state',
                        sources=['baxcat/interface/state.pyx', 'src/state.cpp',
                                 'src/view.cpp', 'src/categorical.cpp',
                                 'src/continuous.cpp', 'src/feature_tree.cpp'],
                        extra_compile_args=['-std=c++11'],
                        extra_link_args=['-lstdc++', '-fopenmp'],
                        include_dirs=['src', 'include', np.get_include(),
                                      boost_dir],
                        language="c++")
             ]

extensions = cythonize(extensions)

setup(
    name='baxcat',
    version='0.0.1',
    author='Baxter S. Eaves Jr.',
    url='TBA',
    long_description='TBA.',
    package_dir={'baxcat': 'baxcat/'},
    ext_modules=extensions,
    cmdclass=cmdclass
)
