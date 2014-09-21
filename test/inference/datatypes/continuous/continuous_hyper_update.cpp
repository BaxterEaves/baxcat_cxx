#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "datatypes/continuous.hpp"
#include "numerics.hpp"
#include "utils.hpp"
#include "test_utils.hpp"

using baxcat::datatypes::Continuous;

const size_t num_samples = 500;


BOOST_AUTO_TEST_SUITE( continuous_hyper_update )

struct TestFeature_single{
    std::vector<double> X;
    std::vector<Continuous> models;

    TestFeature_single(){
        X = {1,2,3,4};
        auto config = Continuous::constructHyperpriorConfig(X);

        models.emplace_back(4, 10, 30, 0, 1.2, 3, 2);
        auto log_fm = Continuous::constructMConditional(models, config);
    }
};

struct TestFeature_multiple{

    std::vector<double> X;

    std::vector<Continuous> models;
    TestFeature_multiple(){
        X = {
            4.87585565178368796069, 6.48969760778546511659, 6.40903448980047940609,
            6.41719241342961410624, -4.70612853290334154366, -5.78728280375863768370, 
            -4.11160436824235819842, -6.14707010696915023829};

        models.emplace_back(4, 20.9748786135057, 119.246738310581, 0, 1.2, 3, 2);
        models.emplace_back(4, -11.967129504214, 51.1789563605734, 0, 1.2, 3, 2);
    }
};

// Single model tests
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(test_m_posterior_update_single)
{
    TestFeature_single feature;

    // m conditionals
    auto config = Continuous::constructHyperpriorConfig(feature.X);

    double w = config[1];
    double x_0 = config[0];
    double ks_stat;

    auto log_fm = Continuous::constructMConditional(feature.models, config);
    ks_stat = baxcat::test_utils::testHyperparameterSampler(log_fm, x_0, {-INF,INF}, w,
        "results/continuous_m_hyper_single.png", "m hypers (s)");
    bool reject_m = baxcat::test_utils::ksTestRejectNull(ks_stat, num_samples, num_samples);

    BOOST_CHECK(!reject_m);
}

BOOST_AUTO_TEST_CASE(test_r_posterior_update_single)
{
    TestFeature_single feature;

    // m conditionals
    auto config = Continuous::constructHyperpriorConfig(feature.X);

    double w = config[2];
    double x_0 = 1/config[2];
    double ks_stat;

    auto log_fr = Continuous::constructRConditional(feature.models, config);    

    ks_stat = baxcat::test_utils::testHyperparameterSampler(log_fr, x_0, {ALMOST_ZERO,10},
        w, "results/continuous_r_hyper_single.png", "r hypers (s)");
    bool reject_r = baxcat::test_utils::ksTestRejectNull(ks_stat, num_samples, num_samples);

    BOOST_CHECK(!reject_r);
}

BOOST_AUTO_TEST_CASE(test_s_posterior_update_single)
{
    TestFeature_single feature;

    // m conditionals
    auto config = Continuous::constructHyperpriorConfig(feature.X);
    auto log_fs = Continuous::constructSConditional(feature.models, config);

    double w = config[3];
    double x_0 = config[3];
    double ks_stat;

    ks_stat = baxcat::test_utils::testHyperparameterSampler(log_fs, x_0, {ALMOST_ZERO,INF}, 
        w, "results/continuous_s_hyper_single.png", "s hypers (s)");
    bool reject_s = baxcat::test_utils::ksTestRejectNull(ks_stat, num_samples, num_samples);

    BOOST_CHECK(!reject_s);
}

BOOST_AUTO_TEST_CASE(test_nu_posterior_update_single)
{
    TestFeature_single feature;

    // m conditionals
    auto config = Continuous::constructHyperpriorConfig(feature.X);
    auto log_fnu = Continuous::constructNuConditional(feature.models);

    double w = 1;
    double x_0 = 1;
    double ks_stat;

    ks_stat = baxcat::test_utils::testHyperparameterSampler(log_fnu, x_0, {ALMOST_ZERO,INF}, 
        w, "results/continuous_nu_hyper_single.png", "nu hypers (s)");
    bool reject_nu = baxcat::test_utils::ksTestRejectNull(ks_stat, num_samples, num_samples);

    BOOST_CHECK(!reject_nu);
}

// Multiple model tests
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(test_m_posterior_update_multiple)
{
    TestFeature_multiple feature;

    // m conditionals
    auto config = Continuous::constructHyperpriorConfig(feature.X);

    double w = config[1];
    double x_0 = config[0];
    double ks_stat;

    auto log_fm = Continuous::constructMConditional(feature.models, config);
    ks_stat = baxcat::test_utils::testHyperparameterSampler(log_fm, x_0, {-INF,INF}, w,
        "results/continuous_m_hyper_multiple.png", "m hypers (m)");
    bool reject_m = baxcat::test_utils::ksTestRejectNull(ks_stat, num_samples, num_samples);

    BOOST_CHECK(!reject_m);
}

BOOST_AUTO_TEST_CASE(test_r_posterior_update_multiple)
{
    TestFeature_multiple feature;

    // m conditionals
    auto config = Continuous::constructHyperpriorConfig(feature.X);

    double w = 1/config[2];
    double x_0 = 1/config[2];
    double ks_stat;

    auto log_fr = Continuous::constructRConditional(feature.models, config);    

    ks_stat = baxcat::test_utils::testHyperparameterSampler(log_fr, x_0, {ALMOST_ZERO,10},
        w, "results/continuous_r_hyper_multiple.png", "r hypers (m)");
    bool reject_r = baxcat::test_utils::ksTestRejectNull(ks_stat, num_samples, num_samples);

    BOOST_CHECK(!reject_r);
}

BOOST_AUTO_TEST_CASE(test_s_posterior_update_multiple)
{
    TestFeature_multiple feature;

    // m conditionals
    auto config = Continuous::constructHyperpriorConfig(feature.X);
    auto log_fs = Continuous::constructSConditional(feature.models, config);

    double w = config[3];
    double x_0 = config[3];
    double ks_stat;

    ks_stat = baxcat::test_utils::testHyperparameterSampler(log_fs, x_0, {ALMOST_ZERO,INF}, 
        w, "results/continuous_s_hyper_multiple.png", "s hypers (m)");
    bool reject_s = baxcat::test_utils::ksTestRejectNull(ks_stat, num_samples, num_samples);

    BOOST_CHECK(!reject_s);
}

BOOST_AUTO_TEST_CASE(test_nu_posterior_update_multiple)
{
    TestFeature_multiple feature;

    // m conditionals
    auto config = Continuous::constructHyperpriorConfig(feature.X);
    auto log_fnu = Continuous::constructNuConditional(feature.models);

    double w = 1;
    double x_0 = 1;
    double ks_stat;

    ks_stat = baxcat::test_utils::testHyperparameterSampler(log_fnu, x_0, {ALMOST_ZERO,INF}, 
        w, "results/continuous_nu_hyper_multiple.png", "\\nu hypers (m)");
    bool reject_nu = baxcat::test_utils::ksTestRejectNull(ks_stat, num_samples, num_samples);

    BOOST_CHECK(!reject_nu);
}


BOOST_AUTO_TEST_SUITE_END()
