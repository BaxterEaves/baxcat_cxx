
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

#ifndef baxcat_cxx_distributions_categorical
#define baxcat_cxx_distributions_categorical

#include <vector>
#include <cmath>
#include "template_helpers.hpp"
#include "utils.hpp"
#include "numerics.hpp"

namespace baxcat{
namespace dist{
namespace categorical{


template <typename T>
static typename baxcat::enable_if<std::is_integral<T>,void> suffstatInsert(T x,
    std::vector<T> &counts)
{
    ++counts[x];
}


template <typename T>
static typename baxcat::enable_if<std::is_integral<T>,void> suffstatRemove(T x,
std::vector<T> &counts)
{
    --counts[x];
}


template <typename T>
static typename baxcat::enable_if<std::is_integral<T>,double> logPdfSuffstats(double n,
    std::vector<T> counts, std::vector<double> p)
{
    double logp = 0;
    for (size_t k = 0; k < counts.size(); ++k)
        logp += counts[k]*log(p[k]);

    return logp;
}


template <typename T>
static typename baxcat::enable_if<std::is_integral<T>,double> logPdf(
    const std::vector<T> &X, std::vector<double> p)
{
    std::vector<T> counts(p.size(),0);
    for(auto &x : X)
        suffstatInsert(x, counts);

    double n = static_cast<double>(baxcat::utils::sum(counts));
    return logPdfSuffstats(n, counts, p);
}

template <typename T>
static typename baxcat::enable_if<std::is_integral<T>,double> logPdf(T x, std::vector<double> p)
{
    std::vector<size_t> counts(p.size(),0);
    counts[x] = 1;
    return logPdfSuffstats(1, counts, p);
}


}}}

#endif
