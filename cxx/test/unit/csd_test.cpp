
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <vector>
#include <iostream>

#include "models/csd.hpp"
#include "distributions/categorical.hpp"
#include "distributions/symmetric_dirichlet.hpp"

#define TOL 10E-8

BOOST_AUTO_TEST_SUITE (categorical_symmetric_dirichlet_test)

using baxcat::models::CategoricalDirichlet;


BOOST_AUTO_TEST_CASE(insert_suffstats_should_add_values)
{
    std::vector<size_t> counts = {0, 0, 0, 0};

    size_t x = 1;
    CategoricalDirichlet<size_t>::suffstatInsert(x, counts);

    BOOST_CHECK_EQUAL(counts[1], 1);
    BOOST_CHECK_EQUAL(counts[0], 0);
    BOOST_CHECK_EQUAL(counts[2], 0);
    BOOST_CHECK_EQUAL(counts[3], 0);

    size_t y = 3;
    CategoricalDirichlet<size_t>::suffstatInsert(y, counts);

    BOOST_CHECK_EQUAL(counts[1], 1);
    BOOST_CHECK_EQUAL(counts[3], 1);
    BOOST_CHECK_EQUAL(counts[0], 0);
    BOOST_CHECK_EQUAL(counts[2], 0);
}


BOOST_AUTO_TEST_CASE(remove_suffstats_should_clear_values)
{
    std::vector<size_t> counts = {1, 3, 2, 0};

    CategoricalDirichlet<size_t>::suffstatRemove(0, counts);

    BOOST_CHECK_EQUAL(counts[0], 0);
    BOOST_CHECK_EQUAL(counts[1], 3);
    BOOST_CHECK_EQUAL(counts[2], 2);
    BOOST_CHECK_EQUAL(counts[3], 0);

    CategoricalDirichlet<size_t>::suffstatRemove(1, counts);
    CategoricalDirichlet<size_t>::suffstatRemove(1, counts);
    CategoricalDirichlet<size_t>::suffstatRemove(1, counts);

    BOOST_CHECK_EQUAL(counts[1], 0);

    CategoricalDirichlet<size_t>::suffstatRemove(2, counts);

    BOOST_CHECK_EQUAL(counts[2], 1);
}


// Should not mutate categorical or dirichlet values
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(log_likelihood_should_be_same_as_dist)
{
    std::vector<size_t> X = {1, 3, 2, 0, 1};
    std::vector<size_t> counts = {1 ,2, 1, 1};
    std::vector<double> P = {.25, .25, .25, .25};

    size_t x = 2;
    double logpdf_categorical = baxcat::dist::categorical::logPdf(X, P);
    double logpdf_msd = CategoricalDirichlet<size_t>::logLikelihood(counts, P);
    double logpdf_categorical_suffstats = baxcat::dist::categorical::logPdfSuffstats(5, counts, P);

    BOOST_CHECK_EQUAL(logpdf_categorical, logpdf_msd);
    BOOST_CHECK_EQUAL(logpdf_categorical_suffstats, logpdf_msd);
}


BOOST_AUTO_TEST_CASE(log_prior_should_be_same_as_dist)
{
    double alpha = 1.4;
    std::vector<double> X = {0.29509583511981724024, 0.32808108145162268032, 0.04599507934921470698,
                             0.33082800407934548348};

    double pdf_sd = baxcat::dist::symmetric_dirichlet::logPdf(X, alpha);
    double pdf_msd = CategoricalDirichlet<size_t>::logPrior(X, alpha);

    BOOST_CHECK_EQUAL(pdf_msd, pdf_sd);
}


// Value checks
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(log_z_value_checks)
{
    double n = 3;
    std::vector<size_t> counts = {1, 1, 1, 1};
    double alpha = 1.45;
    double true_value = 2.17475172148416;

    double msd_value = CategoricalDirichlet<size_t>::logZ(n, counts, alpha);

    BOOST_CHECK_CLOSE_FRACTION(msd_value, true_value, TOL);
}


BOOST_AUTO_TEST_CASE(logPrior_value_checks)
{
    std::vector<double> p = {.2, .3, .5};

    double msd_value;
    msd_value = CategoricalDirichlet<size_t>::logPrior(p, 1.0);

    BOOST_CHECK_CLOSE_FRACTION(0.693147180559945, msd_value, TOL);

    msd_value = CategoricalDirichlet<size_t>::logPrior(p, 2.3);

    BOOST_CHECK_CLOSE_FRACTION(1.37165082501073, msd_value, TOL);
}


BOOST_AUTO_TEST_CASE(logLikelihood_value_checks)
{
    std::vector<double> p = {.2, .3, .5};
    std::vector<size_t> counts = {1, 4, 7};

    double msd_value;
    msd_value =  CategoricalDirichlet<size_t>::logLikelihood(counts, p);
    BOOST_CHECK_CLOSE_FRACTION(-11.277359393657463, msd_value, TOL);
}


BOOST_AUTO_TEST_CASE(logMarginalLikelihood_value_checks)
{
    double n, alpha, msd_value;
    std::vector<size_t> counts;

    n = 10;
    alpha = 1.0;
    counts = {1, 4, 5};

    msd_value =  CategoricalDirichlet<size_t>::logMarginalLikelihood(n, counts, alpha);
    BOOST_CHECK_CLOSE_FRACTION(-11.3285217419719, msd_value, TOL);

    n = 22;
    alpha = .8;
    counts = {2, 7, 13};
    msd_value =  CategoricalDirichlet<size_t>::logMarginalLikelihood(n, counts, alpha);
    BOOST_CHECK_CLOSE_FRACTION(-22.4377193008552, msd_value, TOL);

    alpha = 4.5;
    msd_value =  CategoricalDirichlet<size_t>::logMarginalLikelihood(n, counts, alpha);
    BOOST_CHECK_CLOSE_FRACTION(-22.4203863897293, msd_value, TOL);
}


BOOST_AUTO_TEST_CASE(logPredictiveProbability_value_checks)
{
    double n, alpha, log_z, msd_value;
    std::vector<size_t> counts;

    n = 10;
    alpha = 1.0;
    counts = {1, 4, 5};
    log_z = CategoricalDirichlet<size_t>::logZ(n, counts, alpha);

    msd_value = CategoricalDirichlet<size_t>::logPredictiveProbability(0, counts, alpha, log_z);
    BOOST_CHECK_CLOSE_FRACTION(-1.87180217690159, msd_value, TOL);

    msd_value = CategoricalDirichlet<size_t>::logPredictiveProbability(1, counts, alpha, log_z);
    BOOST_CHECK_CLOSE_FRACTION(-0.95551144502744, msd_value, TOL);

    alpha = 2.5;
    log_z = CategoricalDirichlet<size_t>::logZ(n, counts, alpha);
    msd_value = CategoricalDirichlet<size_t>::logPredictiveProbability(0, counts, alpha, log_z);
    BOOST_CHECK_CLOSE_FRACTION(-1.6094379124341, msd_value, TOL);

    alpha = .25;
    counts = {2, 7, 13};
    log_z = CategoricalDirichlet<size_t>::logZ(n, counts, alpha);
    msd_value = CategoricalDirichlet<size_t>::logPredictiveProbability(0, counts, alpha, log_z);
    BOOST_CHECK_CLOSE_FRACTION(-2.31363492918062, msd_value, TOL);
}


BOOST_AUTO_TEST_CASE(logSingletonProbability_value_checks)
{
    double n, alpha, msd_value;

    // singleton probability should be the same for any x (symmetric dirichlet only)
    alpha = 1;
    msd_value = CategoricalDirichlet<size_t>::logSingletonProbability(0, 3, alpha);
    BOOST_CHECK_CLOSE_FRACTION(-1.09861228866811, msd_value, TOL);

    msd_value = CategoricalDirichlet<size_t>::logSingletonProbability(1, 3, alpha);
    BOOST_CHECK_CLOSE_FRACTION(-1.09861228866811, msd_value, TOL);

    msd_value = CategoricalDirichlet<size_t>::logSingletonProbability(2, 3, alpha);
    BOOST_CHECK_CLOSE_FRACTION(-1.09861228866811, msd_value, TOL);

    // check other alphas
    alpha = 2.5;
    msd_value = CategoricalDirichlet<size_t>::logSingletonProbability(0, 3, alpha);
    BOOST_CHECK_CLOSE_FRACTION(-1.09861228866811, msd_value, TOL);

    alpha = .25;
    msd_value = CategoricalDirichlet<size_t>::logSingletonProbability(0, 3, alpha);
    BOOST_CHECK_CLOSE_FRACTION(-1.09861228866811, msd_value, TOL);
}


BOOST_AUTO_TEST_SUITE_END()
