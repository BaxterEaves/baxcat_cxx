
// BaxCat: an extensible cross-catigorization engine.
// Copyright (C) 2014 Baxter Eaves
//
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License (LICENSE.txt) along with this
// program. If not, see <http://www.gnu.org/licenses/>.
//
// You may contact the mantainers of this software via github
// <https://github.com/BaxterEaves/baxcat_cxx>.

#ifndef baxcat_cxx_constants_guard
#define baxcat_cxx_constants_guard

namespace baxcat{

// Indicate which transition to do
enum transition_type{
    row_assignment,
    column_assignment,
    row_alpha,
    column_alpha,
    column_hypers
};

// Which model to init
enum datatype{
    continuous,     // normal, normal-gamma
    categorical,    // multinomial, dirichlet
    binomial,       // bernoulli, beta
    count,          // poisson, gamma
    magnitude,      // lognormal, normal-gamma
    bounded,        // beta, gamma-exponential
    cyclic,         // von mises, von misses-inverse gamma
};

static std::map<std::string, std::vector<double>> geweke_default_hypers = {
    {"continuous", {0, 1, 10, 10}},
    {"categorical", {1}}
};

static std::map<std::string, std::vector<double>> geweke_default_hyperprior_config = {
    {"continuous", {0, .1, 10, .1, 40, .25, 40, .25}},
    {"categorical", {10}}
};

static std::map<std::string, std::vector<double>> geweke_default_distargs = {
    {"continuous", {0}},
    {"categorical", {5}}
};

static double geweke_default_alpha = 1;

}

#endif
