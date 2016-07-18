
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

#ifndef baxcat_cxx_distributions_symmetric_dirichlet
#define baxcat_cxx_distributions_symmetric_dirichlet

#include <vector>
#include <cmath>
#include "numerics.hpp"

namespace baxcat{
namespace dist{
namespace symmetric_dirichlet{


static double logPdf(std::vector<double> X, double alpha)
{
    size_t K = X.size();
    double A = 0;
    double alpha_minus_one = alpha-1;
    for (auto &x : X)
        A += alpha_minus_one*log(x);

    return -(K*lgamma(alpha)-lgamma(alpha*K))+A;
}


}}}

#endif
