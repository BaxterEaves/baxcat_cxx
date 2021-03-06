
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <memory>
#include "view.hpp"
#include "test_utils.hpp"
#include "datatypes/continuous.hpp"
#include "prng.hpp"
#include "omp.h"


BOOST_AUTO_TEST_SUITE (view_test)

// Using NormalModel Features to test
using std::vector;
using baxcat::Feature;
using baxcat::View;
using baxcat::BaseFeature;
using baxcat::datatypes::Continuous;

struct Setup{
    Setup(baxcat::PRNG *rng){
        baxcat::DataContainer<double> X1({-2,-1,0,1,2});
        f1 = std::shared_ptr<BaseFeature>(new Feature<Continuous, double>(0,X1,vector<double>(),rng));
        baxcat::DataContainer<double> X2({-5,-3,0,3,5});
        f2 = std::shared_ptr<BaseFeature>(new Feature<Continuous, double>(1,X2,vector<double>(),rng));
        baxcat::DataContainer<double> X3({-7,-4,0,4,7});
        f3 = std::shared_ptr<BaseFeature>(new Feature<Continuous, double>(2,X3,vector<double>(),rng));

        features = {f1,f2,f3};
    }

    std::shared_ptr<BaseFeature> f1;
    std::shared_ptr<BaseFeature> f2;
    std::shared_ptr<BaseFeature> f3;

    std::vector<std::shared_ptr<BaseFeature>> features;
};


// constructor test
//`````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(verify_base_constructor){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    Setup s(rng);

    View view( s.features, rng);
    auto K = view.getNumCategories();

    BOOST_CHECK_EQUAL( view.checkPartitions(), 1);

    for(auto &f : s.features)
        BOOST_CHECK_EQUAL(f.get()->getNumClusters(), K);
}

BOOST_AUTO_TEST_CASE(verify_assignment_constructor){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    Setup s(rng);

    double alpha = 14;
    std::vector<size_t> assignment = {0, 0, 1, 1, 1};

    View view( s.features, rng, alpha, assignment);

    BOOST_CHECK_EQUAL(view.checkPartitions(), 1);
    BOOST_CHECK_EQUAL(view.getCRPAlpha(), alpha);
    BOOST_CHECK(baxcat::test_utils::areIdentical(assignment, view.getRowAssignments()));
    BOOST_CHECK(baxcat::test_utils::areIdentical({2,3}, view.getClusterCounts()));
    BOOST_CHECK_EQUAL(view.getNumCategories(), 2);

    for(auto &f : s.features){
        auto suffstats = f.get()->getModelSuffstats();
        size_t n_0 = static_cast<size_t>(suffstats[0]["n"]+.5);
        size_t n_1 = static_cast<size_t>(suffstats[1]["n"]+.5);
        BOOST_CHECK_EQUAL(f.get()->getNumClusters(), 2);
        BOOST_CHECK_EQUAL(n_0, 2);
        BOOST_CHECK_EQUAL(n_1, 3);
    }
}

BOOST_AUTO_TEST_CASE(gibbs_init_should_work){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    Setup s(rng);

    View view( s.features, rng, -1, vector<size_t>(), true);

    BOOST_CHECK_EQUAL(view.checkPartitions(), 1);
}


//`````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(release_feature_should_affect_counts){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    Setup s(rng);

    View view( s.features, rng);

    BOOST_CHECK_EQUAL( view.getNumFeatures(), 3 );

    // release feature 2
    view.releaseFeature( 2 );

    BOOST_CHECK_EQUAL( view.getNumFeatures(), 2 );
}

BOOST_AUTO_TEST_CASE(assimilate_feature_should_affect_counts){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    Setup s(rng);

    View view( s.features, rng );

    BOOST_CHECK_EQUAL( view.getNumFeatures(), 3 );

    // release feature 3
    view.releaseFeature( 2 );

    BOOST_CHECK_EQUAL( view.getNumFeatures(), 2 );

    // assimilate feature 3
    view.assimilateFeature( s.f3 );
    BOOST_CHECK_EQUAL( view.getNumFeatures(), 3 );

    BOOST_CHECK_EQUAL( view.checkPartitions(), 1);
}

BOOST_AUTO_TEST_CASE(get_indices_should_return_feature_indeices){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    Setup s(rng);

    View view( s.features, rng );

    std::vector<size_t> idx = {0,1,2};
    auto indices = view.getFeatureIndices();

    BOOST_CHECK( indices.size() == 3 );
    BOOST_CHECK( baxcat::test_utils::hasSameElements( idx, indices) );
}

// Pop tests
//`````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(pop_row_should_work_non_singleton){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    Setup s(rng);

    std::vector<size_t> assignment_in = {0,0,0,0,0};
    View view( s.features, rng, 1, assignment_in );

    BOOST_REQUIRE_EQUAL(view.getNumRows(), 5);
    BOOST_REQUIRE_EQUAL(view.getNumCategories(), 1);

    auto assignment_a = view.getRowAssignments();
    baxcat::test_utils::areIdentical(assignment_a, assignment_in);

    view.popRow();

    BOOST_REQUIRE_EQUAL(view.getNumRows(), 4);
    BOOST_REQUIRE_EQUAL(view.getNumCategories(), 1);

    auto assignment_b = view.getRowAssignments();
    baxcat::test_utils::areIdentical(assignment_b, {0,0,0,0});
}

BOOST_AUTO_TEST_CASE(pop_row_should_work_non_singleton_1){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    Setup s(rng);

    std::vector<size_t> assignment_in = {1,1,1,1,0};
    View view( s.features, rng, 1, assignment_in );

    BOOST_REQUIRE_EQUAL(view.getNumRows(), 5);
    BOOST_REQUIRE_EQUAL(view.getNumCategories(), 2);

    auto assignment_a = view.getRowAssignments();
    baxcat::test_utils::areIdentical(assignment_a, assignment_in);

    view.popRow();

    BOOST_REQUIRE_EQUAL(view.getNumRows(), 4);
    BOOST_REQUIRE_EQUAL(view.getNumCategories(), 1);

    auto assignment_b = view.getRowAssignments();
    baxcat::test_utils::areIdentical(assignment_b, {0,0,0,0});
}

BOOST_AUTO_TEST_CASE(pop_row_should_work_non_singleton_2){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    Setup s(rng);

    std::vector<size_t> assignment_in = {0,0,0,0,1};
    View view( s.features, rng, 1, assignment_in );

    BOOST_REQUIRE_EQUAL(view.getNumRows(), 5);
    BOOST_REQUIRE_EQUAL(view.getNumCategories(), 2);

    auto assignment_a = view.getRowAssignments();
    baxcat::test_utils::areIdentical(assignment_a, assignment_in);

    view.popRow();

    BOOST_REQUIRE_EQUAL(view.getNumRows(), 4);
    BOOST_REQUIRE_EQUAL(view.getNumCategories(), 1);

    auto assignment_b = view.getRowAssignments();
    baxcat::test_utils::areIdentical(assignment_b, {0,0,0,0});
}

// does't SIGBART tests
//`````````````````````````````````````````````````````````````````````````````
// I'm not entirely sure how to unit test functions that depend on random
// numbers, so I'm just going to run all the functions and see if they SIGBART
// or damage the partitions.
BOOST_AUTO_TEST_CASE(transition_row_should_run){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    Setup s(rng);

    View view( s.features, rng );

    view.transitionRow(0);
    BOOST_CHECK_EQUAL( view.checkPartitions(), 1);

    view.transitionRow(1);
    BOOST_CHECK_EQUAL( view.checkPartitions(), 1);

    view.transitionRow(2);
    BOOST_CHECK_EQUAL( view.checkPartitions(), 1);
}

BOOST_AUTO_TEST_CASE(transition_rows_should_run){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    Setup s(rng);

    View view( s.features, rng );

    view.transitionRows();
    BOOST_CHECK_EQUAL( view.checkPartitions(), 1);
}

BOOST_AUTO_TEST_SUITE_END()
