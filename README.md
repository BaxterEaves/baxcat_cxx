# baxcat_cxx

![Travis-CI status](https://travis-ci.org/BaxterEaves/baxcat_cxx.svg?branch=master)

A C++/python implementation of cross-cutting categorization.

# Installation
C++ backend requires Boost and an OpenMP, C++11 capable compiler.

The python front end requires python version 3.X and the following pacakges:
- cython
- numpy
- scipy
- pandas
- matplotlib
- seaborn
- pytest (for testing)
- freeze (for testing)

To install:

    $ python setup.py install

To run the python unit tests:

    $ py.test

# Currently implemented data types

## Continuous
Uses Normal likelihood with Normal-Gamma prior.

## Categorical
Uses Categorical likelihood with symmetric Dirichlet prior.

## License
Licensed under the GNU General Public License, version 3. See `LICENSE.txt` for details.
