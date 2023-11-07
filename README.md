# baxcat_cxx

A C++/python implementation of cross-cutting categorization.

![Travis-CI status](https://travis-ci.org/BaxterEaves/baxcat_cxx.svg?branch=master)

**baxcat is no longer developed. Please see [lace](https://lace.dev) for all your probabilistic cross-categorization needs**


# Installation
C++ backend requires Boost and an OpenMP, C++11 capable compiler.

The python front end requires python version 3.X and the packages in
`requirements.txt`:

To install:

    $ python setup.py install

To run the python unit tests:

    $ py.test

# Documentation
The documentation is a work in progress. Nevertheless, you can find it
[here](http://baxcat.baxtereaves.com/). Or you can build it by `cd doc && make
html`. The docs will be in _build/html.

# Use
Examples are stored in the `examples` directory. More coming soon.

# Contributing
Follow pep8 with 4-space indents and 79-charcter lines. The default settings on
flake8 should be fine.

# Hacking

Some things I need to do to make this more accessible and useful to non-bax
humans.

- [X] Sphinx documentation
- [ ] Real-world examples in `examples` directory
- [ ] Interval probabilities in `Engine.probability`
- [ ] Count data via Poisson
- [ ] Cyclical data via VonMises
- [ ] Magnitude data via Log-Normal
- [X] Row similarity heatmap
- [X] Row similarity WRT specific columns (#1)
- [X] Way to evalute predictive power `Engine.eval`
- [X] Optional ouput during Engine.run
- [X] More options for parallelization (IPython, Pathos, etc)

Optimizations and refactoring

- [ ] Should have `Engine._converters['valmaps']` for every column
- [X] rename `msd` in `cxx` to `csd` (#3) 
- [X] Remove stupid copyright boilerplate.
- [ ] dial back the namespaces and better organize headers
- [X] Figure out better prior for CRP alpha. Maybe need prior MH to avoid
  numerical problems (#2)
- [X] Fix super-redundant code in `Engine.pairwise_func`
- [ ] Refactor DataContainer

# License
BaxCat: an extensible cross-catigorization engine.
Copyright (C) 2016 Baxter Eaves

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, version 3.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License (LICENSE.txt)
along with this program. If not, see <http://www.gnu.org/licenses/>.

You may contact the mantainers of this software via github <https://github.com/BaxterEaves/baxcat_cxx>.
