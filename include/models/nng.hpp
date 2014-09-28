
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

#ifndef baxcat_cxx_datamodels_nng
#define baxcat_cxx_datamodels_nng

#include "debug.hpp"

#include <cmath>
#include <vector>
#include <iostream>

#include "prng.hpp"
#include "distributions/gamma.hpp"
#include "distributions/gaussian.hpp"
#include "distributions/students_t.hpp"


namespace baxcat{
namespace models{

struct NormalNormalGamma{

    // NOTE: ZN, the marginal probability normalizing constant needs only be updated when the
    // sufficient statistics are updated. We could cache ZN and use it to calculate predictive
    // probability more quickly (but we're not currently taking advantage of it)

    // Add/remove data
    //`````````````````````````````````````````````````````````````````````````````````````````
    static void suffstatInsert(double x, double &sum_x, double &sum_x_sq)
    {
        baxcat::dist::gaussian::suffstatInsert(x, sum_x, sum_x_sq);
    }

    static void suffstatRemove(double x, double &sum_x, double &sum_x_sq)
    {
        baxcat::dist::gaussian::suffstatRemove(x, sum_x, sum_x_sq);
    }

    // quantities for testing and uncollapsed samplers
    //`````````````````````````````````````````````````````````````````````````````````````````
    static double logLikelihood(double n, double sum_x, double sum_x_sq, double mu, double rho)
    {
        return baxcat::dist::gaussian::logPdfSuffstats(n, sum_x, sum_x_sq, mu, rho);
    }

    static double logPrior(double mu, double rho, double m, double r, double s, double nu)
    {
        double logp_mu = baxcat::dist::gaussian::logPdf(mu, m, rho/r);
        double logp_rho = baxcat::dist::gamma::logPdf(rho, nu/2, 1/s);

        ASSERT_IS_A_NUMBER(std::cout, logp_mu);
        ASSERT_IS_A_NUMBER(std::cout, logp_rho);

        return logp_mu+logp_rho;
    }

    // Updaters
    //`````````````````````````````````````````````````````````````````````````````````````````
    static double logZ(double r, double s, double nu)
    {
        return ((nu+1)/2)*LOG_2 + .5*LOG_PI - .5*log(r) - .5*nu*log(s) + lgamma(nu/2);
    }

    static void posteriorParameters(double n, double sum_x, double sum_x_sq, double &m_n,
        double &r_n, double &s_n, double &nu_n)
    {
        double r = r_n;
        double nu = nu_n;
        double m = m_n;
        double s = s_n;

        r_n = r + n;
        nu_n = nu + n;
        m_n = (r*m+sum_x)/(r+n);
        s_n = s + sum_x_sq + r*(m*m) - r_n*(m_n*m_n);
    }

    // Marginal Likelihood
    //`````````````````````````````````````````````````````````````````````````````````````````
    // with cached log_ZN
    static double logMarginalLikelihood(double n, double log_ZN, double log_Z0)
    {
        return -(n/2)*LOG_2PI + log_ZN-log_Z0;
    }

    // with cached log_Z0
    static double logMarginalLikelihood(double n, double sum_x, double sum_x_sq, double m,
        double r, double s, double nu, double log_Z0)
    {
        posteriorParameters(n, sum_x, sum_x_sq, m, r, s, nu);
        double log_ZN = logZ(r, s, nu);
        return logMarginalLikelihood(n, log_ZN, log_Z0);
    }

    // no caching (needed for hyperparameter conditionals)
    static double logMarginalLikelihood(double n, double sum_x, double sum_x_sq, double m,
        double r, double s, double nu)
    {
        // double log_Z0 = logZ(r, s, nu);
        return logMarginalLikelihood(n, sum_x, sum_x_sq, m, r, s, nu, logZ(r, s, nu));
    }

    // Posterior Predictive
    //`````````````````````````````````````````````````````````````````````````````````````````
    // with cached log_ZN
    static double logPredictiveProbability(double x, double n, double sum_x, double sum_x_sq,
        double m, double r, double s, double nu, double log_ZN)
    {

        double r_m = r;
        double nu_m = nu;
        double m_m = m;
        double s_m = s;

        posteriorParameters(n+1, sum_x+x, sum_x_sq+x*x, m_m, r_m, s_m, nu_m);
        double log_ZM = logZ(r_m, s_m, nu_m);

        return -.5*LOG_2PI + log_ZM-log_ZN;
    }

    static double logPredictiveProbability(double x, double n, double sum_x, double sum_x_sq,
        double m, double r, double s, double nu)
    {
        double r_n = r;
        double nu_n = nu;
        double m_n = m;
        double s_n = s;

        posteriorParameters(n, sum_x, sum_x_sq, m_n, r_n, s_n, nu_n);
        double log_ZN = logZ(r_n, s_n, nu_n);

        return  logPredictiveProbability(x, n, sum_x, sum_x_sq, m, r, s, nu, log_ZN);
    }

    // Sampling
    //`````````````````````````````````````````````````````````````````````````````````````````
    static double predictiveSample(double n, double sum_x, double sum_x_sq, double m, double r,
        double s, double nu, baxcat::PRNG *rng)
    {

        ASSERT_GREATER_THAN_ZERO(std::cout, r);
        ASSERT_GREATER_THAN_ZERO(std::cout, s);
        ASSERT_GREATER_THAN_ZERO(std::cout, nu);
        ASSERT_IS_A_NUMBER(std::cout, m);
        ASSERT_IS_A_NUMBER(std::cout, r);
        ASSERT_IS_A_NUMBER(std::cout, s);
        ASSERT_IS_A_NUMBER(std::cout, nu);

        if( n > 0)
            posteriorParameters(n, sum_x, sum_x_sq, m, r, s, nu);

        s /= 2;

        double t_draw = rng->trand(nu);

        ASSERT_IS_A_NUMBER(std::cout, t_draw);

        double scale = sqrt((s * (r + 1)) / (nu / 2. * r));

        ASSERT_IS_A_NUMBER(std::cout, scale);

        return t_draw*scale + m;
    }

    // TODO: Fill this in
    static void posteriorSample(double n, double sum_x, double sum_x_sq, double m, double r,
        double s, double nu, double &mu, double &rho, baxcat::PRNG *rng)
    {
        posteriorParameters(n, sum_x, sum_x_sq, m, r, s, nu);
        rho = rng->gamrand(nu/2, 1/s);
        mu = rng->normrand(m, rho/r);
    }

};
}} // end namespaces

#endif
