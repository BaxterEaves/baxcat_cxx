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

# Use
Examples are stored in the `examples` directory. More coming soon.

# Currently implemented data types

## Continuous
Uses Normal likelihood with Normal-Gamma prior.

## Categorical
Uses Categorical likelihood with symmetric Dirichlet prior.

# Hacking

Some things I need to do to make this more accessible and useful to non-bax humans.

- [ ] Spynx documentation
- [ ] Real-world examples in `examples` directory
- [ ] Interval probabilities in `Engine.probability`
- [X] Row similarity heatmap
- [ ] Row similarity WRT specific columns (#1)
- [X] Way to evalute predictive power `Engine.eval`
- [X] Optional ouput during Engine.run

Optimizations and refactoring

- [ ] Should have `Engine._converters['valmaps']` for every column
- [X] rename `msd` in `cxx` to `csd` (#3) 
- [ ] Remove stupid copyright boilerplate.
- [ ] dial back the namespaces and better organize headers
- [ ] Figure out better prior for CRP alpha. Maybe need prior MH to avoid numerical problems (#2)
- [X] Fix super-redundant code in `Engine.pairwise_func`

# License
Licensed under the GNU General Public License, version 3. See `LICENSE.txt` for details.
