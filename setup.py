from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from pip.req import parse_requirements

import numpy as np
import os

here = os.path.dirname(os.path.realpath(__file__))
reqfile = os.path.join(here, 'requirements.txt')

install_reqs = parse_requirements(reqfile, session=False)
reqs = [str(ir.req) for ir in install_reqs]

SRC = os.path.join('cxx', 'src')
INC = os.path.join('cxx', 'include')

baxcat_compile_args = ['-std=c++11', '-Wno-comment', '-fopenmp']
readthedocs = os.environ.get('READTHEDOCS', None) == 'True'

boost_url = "https://sourceforge.net/projects/boost/files/boost/1.61.0/"\
            "boost_1_61_0.tar.gz"


class BaxcatInstall(build_ext):
    def run(self):
        print("Yo.")
        if readthedocs:
            import urllib
            print("Downloading boost")
            urllib.request.urlretrieve(boost_url, "boost_1_61_0.tar.gz")
            print("Extracting BOOST to %s" % (here,))
            os.system("tar -zxf boost_1_61_0.tar.gz")
            print("Adding to compile args")
            baxcat_compile_args.append('-I ' + os.path.join(here, 'boost_1_61_0/'))
        build_ext.run(self)

extensions = [
    Extension('baxcat.state',
              sources=[os.path.join('baxcat', 'interface', 'state.pyx'),
                       os.path.join(SRC, 'state.cpp'),
                       os.path.join(SRC, 'view.cpp'),
                       os.path.join(SRC, 'categorical.cpp'),
                       os.path.join(SRC, 'continuous.cpp'),
                       os.path.join(SRC, 'feature_tree.cpp')],
              extra_compile_args=baxcat_compile_args,
              extra_link_args=['-lstdc++', '-fopenmp'],
              include_dirs=[SRC, INC, np.get_include()],
              language="c++"),
    Extension('baxcat.dist.nng',
              sources=[os.path.join('baxcat', 'dist', 'nng.pyx')],
              extra_compile_args=['-std=c++11', '-fopenmp'],
              extra_link_args=['-lstdc++', '-fopenmp'],
              include_dirs=[INC, np.get_include()],
              language="c++"),
    Extension('baxcat.dist.csd',
              sources=[os.path.join('baxcat', 'dist', 'csd.pyx')],
              extra_compile_args=['-std=c++11', '-fopenmp'],
              extra_link_args=['-lstdc++', '-fopenmp'],
              include_dirs=[INC, np.get_include()],
              language="c++")]

setup(
    name='baxcat',
    version='0.2a1.dev1',
    author='Baxter S. Eaves Jr.',
    url='TBA',
    long_description='TBA.',
    package_dir={'baxcat': 'baxcat/'},
    setup_requires=reqs,
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': BaxcatInstall}
)
