
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "test_utils.hpp"

using std::vector;
using std::string;


BOOST_AUTO_TEST_SUITE( continuous_inference_single )

BOOST_AUTO_TEST_CASE(continuous_1_column_3_cluster_high_separation)
{
    bool distributions_differ;
    distributions_differ = baxcat::test_utils::testOneDimInfereneceQuality(500,3,500,.9,
        "continuous", "results/continuous_1col_3clu_highsep.png");

    BOOST_CHECK(!distributions_differ);
}

BOOST_AUTO_TEST_CASE(continuous_1_column_3_cluster_medium_separation)
{
    bool distributions_differ;
    distributions_differ = baxcat::test_utils::testOneDimInfereneceQuality(500,3,500,.65,
        "continuous", "results/continuous_1col_3clu_midsep.png");

    BOOST_CHECK(!distributions_differ);
}

BOOST_AUTO_TEST_CASE(continuous_1_column_3_cluster_low_separation)
{
    bool distributions_differ;
    distributions_differ = baxcat::test_utils::testOneDimInfereneceQuality(500,3,500,.4,
        "continuous", "results/continuous_1col_3clu_lowsep.png");

    BOOST_CHECK(!distributions_differ);
}

BOOST_AUTO_TEST_CASE(continuous_1_column_3_cluster_no_separation)
{
    bool distributions_differ;
    distributions_differ = baxcat::test_utils::testOneDimInfereneceQuality(500,3,500,0.0,
        "continuous", "results/continuous_1col_3clu_nosep.png");

    BOOST_CHECK(!distributions_differ);
}


BOOST_AUTO_TEST_SUITE_END()

