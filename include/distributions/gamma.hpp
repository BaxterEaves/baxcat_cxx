
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

#ifndef baxcat_cxx_distributions_gamma
#define baxcat_cxx_distributions_gamma

#include <boost/math/distributions/gamma.hpp>
#include <vector>
#include <cmath>
#include "numerics.hpp"

namespace baxcat {
namespace dist {
namespace gamma{


static double logPdf(double x, double shape, double scale)
{
    return -lgamma(shape)-shape*log(scale) + (shape-1)*log(x)-(x/scale);
}


static double logPdf(std::vector<double> X, double shape, double scale)
{
    double logp = 0;
    for (auto &x : X)
        logp += logPdf(x, shape, scale);

    return logp;
}


static double cdf(double x, double shape, double scale)
{
    boost::math::gamma_distribution<> dist(shape, scale);
    return cdf(dist, x);
}


}}}

#endif
