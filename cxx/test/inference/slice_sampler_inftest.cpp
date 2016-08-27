
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <vector>
#include <limits>

#include "prng.hpp"
#include "utils.hpp"
#include "numerics.hpp"
#include "test_utils.hpp"
#include "samplers/slice.hpp"
#include "distributions/beta.hpp"
#include "distributions/gamma.hpp"
#include "distributions/gaussian.hpp"
#include "distributions/inverse_gamma.hpp"

const size_t NUM_SAMPLES = 500;
const size_t LAG = 25;


BOOST_AUTO_TEST_SUITE( slice_sampler_inference_test )

BOOST_AUTO_TEST_CASE(test_slice_sampler_standard_normal)
{
    double x_0 = 0;
    double mu = 0;
    double rho = 1;
    double ks_stat;

    auto log_norm_pdf = [mu, rho](double x){ return baxcat::dist::gaussian::logPdf(x,mu,rho);};
    auto norm_cdf = [mu, rho](double x){ return baxcat::dist::gaussian::cdf(x,mu,rho); };
    baxcat::Domain D(-INF, INF);

    ks_stat = baxcat::test_utils::testSliceSampler(x_0, log_norm_pdf, D, 1, norm_cdf, NUM_SAMPLES);

    bool norm_reject = baxcat::test_utils::ksTestRejectNull(ks_stat, NUM_SAMPLES, NUM_SAMPLES);

    BOOST_CHECK(!norm_reject);
}

BOOST_AUTO_TEST_CASE(test_slice_sampler_beta)
{
    double x_0 = .5;
    double a = 5;
    double b = 1;
    double ks_stat;

    auto log_beta_pdf = [a, b](double x){
        return baxcat::dist::beta::logPdf(x,a,b);
    };
    auto beta_cdf = [a, b](double x){
        return baxcat::dist::beta::cdf(x,a,b);
    };
    baxcat::Domain D(0, 1);

    ks_stat = baxcat::test_utils::testSliceSampler(x_0, log_beta_pdf, D, 1, beta_cdf, NUM_SAMPLES);

    bool beta_reject = baxcat::test_utils::ksTestRejectNull(ks_stat, NUM_SAMPLES, NUM_SAMPLES);
    BOOST_CHECK(!beta_reject);
}

BOOST_AUTO_TEST_CASE(test_slice_sampler_gamma)
{
    double x_0 = 1;
    double shape = 5;
    double scale = 1;
    double ks_stat;

    auto log_gam_pdf = [shape, scale](double x){
        return baxcat::dist::gamma::logPdf(x,shape,scale);
    };
    auto gam_cdf = [shape, scale](double x){
        return baxcat::dist::gamma::cdf(x,shape,scale);
    };
    baxcat::Domain D(0, INF);

    ks_stat = baxcat::test_utils::testSliceSampler(x_0, log_gam_pdf, D, 1, gam_cdf, NUM_SAMPLES);

    bool gamma_reject = baxcat::test_utils::ksTestRejectNull(ks_stat, NUM_SAMPLES, NUM_SAMPLES);
    BOOST_CHECK(!gamma_reject);
}

BOOST_AUTO_TEST_CASE(test_slice_sampler_inverse_gamma)
{
    double x_0 = 1;
    double shape = 2;
    double scale = 3;
    double ks_stat;

    auto log_invgam_pdf = [shape, scale](double x){
        return baxcat::dist::inverse_gamma::logPdf(x,shape,scale);
    };
    auto invgam_cdf = [shape, scale](double x){
        return baxcat::dist::inverse_gamma::cdf(x,shape,scale);
    };
    baxcat::Domain D(0, INF);

    ks_stat = baxcat::test_utils::testSliceSampler(x_0, log_invgam_pdf, D, 1, invgam_cdf, NUM_SAMPLES);

    bool invgamma_reject = baxcat::test_utils::ksTestRejectNull(ks_stat, NUM_SAMPLES, NUM_SAMPLES);
    BOOST_CHECK(!invgamma_reject);
}

BOOST_AUTO_TEST_CASE(test_slice_sampler_normal_mixture)
{
    double x_0 = 0;
    double mu_1 = -3;
    double mu_2 = 3;
    double rho = 1;
    double ks_stat;

    auto log_norm_pdf = [mu_1, mu_2, rho](double x){
        double logp_1 = log(.5) + baxcat::dist::gaussian::logPdf(x,mu_1,rho);
        double logp_2 = log(.5) + baxcat::dist::gaussian::logPdf(x,mu_2,rho);
        double log_pdf_val = baxcat::numerics::logsumexp({logp_1, logp_2});
        return log_pdf_val;
    };
    auto norm_cdf = [mu_1, mu_2, rho](double x){
        double cdf_1 = .5*baxcat::dist::gaussian::cdf(x,mu_1,rho);
        double cdf_2 = .5*baxcat::dist::gaussian::cdf(x,mu_2,rho);
        return cdf_1+cdf_2;
    };

    baxcat::Domain D(-INF, INF);

    ks_stat = baxcat::test_utils::testSliceSampler(x_0, log_norm_pdf, D, 1, norm_cdf, NUM_SAMPLES);

    bool nm_reject = baxcat::test_utils::ksTestRejectNull(ks_stat, NUM_SAMPLES, NUM_SAMPLES);
    BOOST_CHECK(!nm_reject);
}

BOOST_AUTO_TEST_CASE(test_slice_sampler_crp)
{
    auto log_crp = [](double x){
        double logp = baxcat::numerics::lcrpUNormPost(5, 100, x);
        return logp + baxcat::dist::inverse_gamma::logPdf(x,1,1);
    };

    double w = 1;
    double x_0 = 1;
    double ks_stat;

    ks_stat = baxcat::test_utils::testHyperparameterSampler(log_crp, x_0, {0,INF}, w,
        "crp_alpha.png", "CRP \\alpha");

    bool crp_reject = baxcat::test_utils::ksTestRejectNull(ks_stat, NUM_SAMPLES, NUM_SAMPLES);
    BOOST_CHECK(!crp_reject);
}

BOOST_AUTO_TEST_SUITE_END()
