
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <vector>
#include <iostream>

#include "models/nng.hpp"
#include "distributions/gamma.hpp"
#include "distributions/gaussian.hpp"


BOOST_AUTO_TEST_SUITE (normal_normal_gamma_test)

using baxcat::models::NormalNormalGamma;

BOOST_AUTO_TEST_CASE(insert_suffstats_should_add_values){
    double sum_x = 0;
    double sum_x_sq = 0;

    double x = 2;
    NormalNormalGamma::suffstatInsert(x, sum_x, sum_x_sq);

    BOOST_CHECK_EQUAL(sum_x, x);
    BOOST_CHECK_EQUAL(sum_x_sq, x*x);

    double y = 2.5;
    NormalNormalGamma::suffstatInsert(y, sum_x, sum_x_sq);

    BOOST_CHECK_EQUAL(sum_x, x+y);
    BOOST_CHECK_EQUAL(sum_x_sq, x*x + y*y);
}

BOOST_AUTO_TEST_CASE(remove_suffstats_should_clear_values){
    double x = 2;
    double y = 2.5;
    double sum_x = x+y;
    double sum_x_sq = x*x + y*y;

    NormalNormalGamma::suffstatRemove(x, sum_x, sum_x_sq);
    NormalNormalGamma::suffstatRemove(y, sum_x, sum_x_sq);

    BOOST_CHECK_SMALL(sum_x, TOL);
    BOOST_CHECK_SMALL(sum_x_sq, TOL);
}

BOOST_AUTO_TEST_CASE(log_likelihood_should_be_the_same_as_dist){
    double n = 4;
    double sum_x = 1 + 2 + 3 + 4;
    double sum_x_sq = 1*1 + 2*2 + 3*3 + 4*4;

    double mu = 2.5;
    double rho = 2.0;

    double nng_log_likelihood = NormalNormalGamma::logLikelihood(n, sum_x, sum_x_sq, mu, rho);
    double gauss_log_likelihood = baxcat::dist::gaussian::logPdfSuffstats(n, sum_x, sum_x_sq, mu,
                                                                          rho);

    BOOST_CHECK_EQUAL(nng_log_likelihood, gauss_log_likelihood);

}

BOOST_AUTO_TEST_CASE(log_prior_should_be_the_same_as_dist){
    double mu = 0;
    double rho = 1.2;
    double m = .1;
    double r = 1;
    double s = 2;
    double nu = 2;

    double logp_mu = baxcat::dist::gaussian::logPdf(mu, m, rho/r);
    double logp_rho = baxcat::dist::gamma::logPdf(rho, nu/2, 1/s);

    double log_prior = NormalNormalGamma::logPrior(mu, rho, m, r, s, nu);

    BOOST_CHECK_EQUAL(log_prior, logp_mu+logp_rho);
}

// log Z value checks
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(log_Z_value_checks){
    double r = 1;
    double s = 1;
    double nu = 1;

    BOOST_CHECK_CLOSE_FRACTION( NormalNormalGamma::logZ(r,s,nu), 1.83787706640935, TOL);

    r = 1.2;
    s = .4;
    nu = 5.2;

    BOOST_CHECK_CLOSE_FRACTION( NormalNormalGamma::logZ(r,s,nu), 5.36972819068534, TOL);
}


// marginal and predictive probability value checks (no chaching)
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(marginal_likelihood_value_checks){
    double n = 4;
    double sum_x = 10;
    double sum_x_sq = 30;
    double m = 2.1;
    double r = 1.2;
    double s = 1.3;
    double nu = 1.4;

    double log_z = NormalNormalGamma::logZ(r,s,nu);

    double log_nng_ml = NormalNormalGamma::logMarginalLikelihood(n, sum_x, sum_x_sq, m, r, s, nu,
                                                                   log_z);

    BOOST_CHECK_CLOSE_FRACTION( log_nng_ml, -7.69707018344038, TOL);
}

BOOST_AUTO_TEST_CASE(predictive_probability_value_checks){
    double n = 4;
    double sum_x = 10;
    double sum_x_sq = 30;
    double m = 2.1;
    double r = 1.2;
    double s = 1.3;
    double nu = 1.4;
    double log_nng_pp;

    log_nng_pp = NormalNormalGamma::logPredictiveProbability(3, n, sum_x, sum_x_sq, m, r, s, nu);

    BOOST_CHECK_CLOSE_FRACTION( log_nng_pp, -1.28438638499611, TOL);

    log_nng_pp = NormalNormalGamma::logPredictiveProbability(-3, n, sum_x, sum_x_sq, m, r, s, nu);

    BOOST_CHECK_CLOSE_FRACTION( log_nng_pp, -6.1637698862186, TOL);

}

// marginal and predictive probability value checks (with chaching)
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(marginal_likelihood_cached_value_checks){
    double n = 4;
    double sum_x = 10;
    double sum_x_sq = 30;
    double m = 2.1;
    double r = 1.2;
    double s = 1.3;
    double nu = 1.4;

    double log_z = NormalNormalGamma::logZ(r,s,nu);

    NormalNormalGamma::posteriorParameters(n, sum_x, sum_x_sq, m, r, s, nu);

    double log_zn = NormalNormalGamma::logZ(r,s,nu);

    double log_nng_ml = NormalNormalGamma::logMarginalLikelihood(n, log_zn, log_z);

    BOOST_CHECK_CLOSE_FRACTION( log_nng_ml, -7.69707018344038, TOL);
}

BOOST_AUTO_TEST_CASE(predictive_probability_cached_value_checks){
    double n = 4;
    double sum_x = 10;
    double sum_x_sq = 30;
    double m = 2.1;
    double r = 1.2;
    double s = 1.3;
    double nu = 1.4;
    double log_nng_pp;

    double m_n = m;
    double r_n = r;
    double s_n = s;
    double nu_n = nu;

    NormalNormalGamma::posteriorParameters(n, sum_x, sum_x_sq, m_n, r_n, s_n, nu_n);
    double log_zn = NormalNormalGamma::logZ(r_n, s_n, nu_n);

    log_nng_pp = NormalNormalGamma::logPredictiveProbability(3, n, sum_x, sum_x_sq, m, r, s, nu,
                                                               log_zn);

    BOOST_CHECK_CLOSE_FRACTION( log_nng_pp, -1.28438638499611, TOL);

    log_nng_pp = NormalNormalGamma::logPredictiveProbability(-3, n, sum_x, sum_x_sq, m, r, s, nu,
                                                              log_zn);

    BOOST_CHECK_CLOSE_FRACTION( log_nng_pp, -6.1637698862186, TOL);

}

BOOST_AUTO_TEST_SUITE_END()
