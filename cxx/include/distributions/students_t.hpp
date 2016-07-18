
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

#ifndef baxcat_cxx_distributions_students_t
#define baxcat_cxx_distributions_students_t

#include <vector>
#include <random>
#include <cmath>
#include "numerics.hpp"

namespace baxcat{
namespace dist{
namespace students_t{


static double logPdf(double x, double df)
{
    return lgamma((df+1)/2)-lgamma(df/2)-.5*log(df*M_PI)-((df+1)/2)*log(1+x*x/df);
}


static double logPdf(std::vector<double> X, double df)
{
    double logp = 0;
    for (auto &x : X)
        logp += logPdf(x, df);

    return logp;
}


}}}

#endif
