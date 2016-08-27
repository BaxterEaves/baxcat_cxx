
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>

#include "helpers/synthetic_data_generator.hpp"
#include "test_utils.hpp"
#include "utils.hpp"


BOOST_AUTO_TEST_SUITE (test_synthetic_data_generator)

using std::vector;
using std::string;


// Constructors
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(verify_constructor_zero_one_column)
{
    size_t num_rows = 10;
    vector<string> datatypes = {"continuous"};
    size_t seed = 10;

    baxcat::SyntheticDataGenerator sdg(num_rows, datatypes, seed);

    auto data = sdg.getData();

    BOOST_REQUIRE_EQUAL( data.size(), 1 );
    BOOST_REQUIRE_EQUAL( data[0].size(), num_rows );
}


BOOST_AUTO_TEST_CASE(verify_constructor_zero_two_column)
{
    size_t num_rows = 10;
    vector<string> datatypes = {"continuous","continuous"};
    size_t seed = 10;

    baxcat::SyntheticDataGenerator sdg(num_rows, datatypes, seed);

    auto data = sdg.getData();

    BOOST_REQUIRE_EQUAL( data.size(), 2 );
    BOOST_REQUIRE_EQUAL( data[0].size(), num_rows );
    BOOST_REQUIRE_EQUAL( data[1].size(), num_rows );
}


BOOST_AUTO_TEST_CASE(verify_constructor_one_view_one_cluster_continuous)
{
    size_t num_rows = 10;
    vector<double> view_weights = {1};
    vector<vector<double>> category_weights = {{1}};
    vector<double> category_separation = {.9}; // one column
    vector<string> datatypes = {"continuous"};
    size_t seed = 10;


    baxcat::SyntheticDataGenerator sdg(num_rows, view_weights, category_weights,
        category_separation, datatypes, seed);

    auto data = sdg.getData();

    BOOST_REQUIRE_EQUAL( data.size(), 1 );
    BOOST_REQUIRE_EQUAL( data[0].size(), num_rows );
}


BOOST_AUTO_TEST_CASE(verify_constructor_two_columns_continuous)
{
    size_t num_rows = 10;
    vector<double> view_weights = {1};
    vector<vector<double>> category_weights = {{1}};
    vector<double> category_separation = {.9,.9}; // two columns
    vector<string> datatypes = {"continuous","continuous"};
    size_t seed = 10;


    baxcat::SyntheticDataGenerator sdg(num_rows, view_weights, category_weights,
        category_separation, datatypes, seed);

    auto data = sdg.getData();

    BOOST_REQUIRE_EQUAL( data.size(), 2 );
    BOOST_REQUIRE_EQUAL( data[0].size(), num_rows );
    BOOST_REQUIRE_EQUAL( data[1].size(), num_rows );
}


BOOST_AUTO_TEST_CASE(verify_constructor_one_view_two_cluster_continuous)
{
    size_t num_rows = 10;
    vector<double> view_weights = {1};
    vector<vector<double>> category_weights = {{.5,.5}};
    vector<double> category_separation = {.9}; // one column
    vector<string> datatypes = {"continuous"};
    size_t seed = 10;


    baxcat::SyntheticDataGenerator sdg(num_rows, view_weights, category_weights,
        category_separation, datatypes, seed);

    auto data = sdg.getData();

    BOOST_REQUIRE_EQUAL( data.size(), 1 );
    BOOST_REQUIRE_EQUAL( data[0].size(), num_rows );
}


BOOST_AUTO_TEST_CASE(verify_constructor_two_views_two_cluster_continuous)
{
    size_t num_rows = 10;
    vector<double> view_weights = {.5,.5};
    vector<vector<double>> category_weights = {{.5,.5},{.5,.5}};
    vector<double> category_separation = {.9,.9}; // two columns
    vector<string> datatypes = {"continuous","continuous"};
    size_t seed = 10;


    baxcat::SyntheticDataGenerator sdg(num_rows, view_weights, category_weights,
        category_separation, datatypes, seed);

    auto data = sdg.getData();

    BOOST_REQUIRE_EQUAL( data.size(), 2 );
    BOOST_REQUIRE_EQUAL( data[0].size(), num_rows );
    BOOST_REQUIRE_EQUAL( data[1].size(), num_rows );
}


// Weighted partition function
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(verify_partition_generator)
{
    size_t num_rows = 10;
    vector<double> view_weights = {1};
    vector<vector<double>> category_weights = {{1}};
    vector<double> category_separation = {.9}; // one column
    vector<string> datatypes = {"continuous"};
    size_t seed = 10;

    baxcat::SyntheticDataGenerator sdg(num_rows, view_weights, category_weights,
        category_separation, datatypes, seed);

    // generate two even partitions
    auto partition = sdg.generateWeightedParition({.5,.5}, 10);

    BOOST_REQUIRE_EQUAL( partition.size(), 10 );
    BOOST_REQUIRE_EQUAL( baxcat::utils::vector_max(partition), 1 );
    BOOST_REQUIRE_EQUAL( baxcat::utils::vector_min(partition), 0 );

    // three partitions
    partition = sdg.generateWeightedParition({.25,.25,.5}, 10);
    BOOST_REQUIRE_EQUAL( partition.size(), 10 );
    BOOST_REQUIRE_EQUAL( baxcat::utils::vector_max(partition), 2 );
    BOOST_REQUIRE_EQUAL( baxcat::utils::vector_min(partition), 0 );

    // fringe case: 1 partition
    partition = sdg.generateWeightedParition({1}, 10);
    BOOST_REQUIRE_EQUAL( partition.size(), 10 );
    BOOST_REQUIRE_EQUAL( baxcat::utils::vector_max(partition), 0 );
    BOOST_REQUIRE_EQUAL( baxcat::utils::vector_min(partition), 0 );

}


// Categorical
BOOST_AUTO_TEST_CASE(verify_constructor_one_view_one_cluster_categorical)
{
    size_t num_rows = 10;
    vector<double> view_weights = {1};
    vector<vector<double>> category_weights = {{1}};
    vector<double> category_separation = {.9}; // one column
    vector<string> datatypes = {"categorical"};
    size_t seed = 10;


    baxcat::SyntheticDataGenerator sdg(num_rows, view_weights, category_weights,
        category_separation, datatypes, seed);

    auto data = sdg.getData();

    BOOST_REQUIRE_EQUAL( data.size(), 1 );
    BOOST_REQUIRE_EQUAL( data[0].size(), num_rows );
}


BOOST_AUTO_TEST_CASE(verify_constructor_two_columns_categorical)
{
    size_t num_rows = 10;
    vector<double> view_weights = {1};
    vector<vector<double>> category_weights = {{1}};
    vector<double> category_separation = {.9,.9}; // two columns
    vector<string> datatypes = {"categorical","categorical"};
    size_t seed = 10;


    baxcat::SyntheticDataGenerator sdg(num_rows, view_weights, category_weights,
        category_separation, datatypes, seed);

    auto data = sdg.getData();

    BOOST_REQUIRE_EQUAL( data.size(), 2 );
    BOOST_REQUIRE_EQUAL( data[0].size(), num_rows );
    BOOST_REQUIRE_EQUAL( data[1].size(), num_rows );
}


BOOST_AUTO_TEST_CASE(verify_constructor_one_view_two_cluster_categorical)
{
    size_t num_rows = 10;
    vector<double> view_weights = {1};
    vector<vector<double>> category_weights = {{.5,.5}};
    vector<double> category_separation = {.9}; // one column
    vector<string> datatypes = {"categorical"};
    size_t seed = 10;


    baxcat::SyntheticDataGenerator sdg(num_rows, view_weights, category_weights,
        category_separation, datatypes, seed);

    auto data = sdg.getData();

    BOOST_REQUIRE_EQUAL( data.size(), 1 );
    BOOST_REQUIRE_EQUAL( data[0].size(), num_rows );
}


BOOST_AUTO_TEST_CASE(verify_constructor_two_views_two_cluster_categorical)
{
    size_t num_rows = 10;
    vector<double> view_weights = {.5,.5};
    vector<vector<double>> category_weights = {{.5,.5},{.5,.5}};
    vector<double> category_separation = {.9,.9}; // two columns
    vector<string> datatypes = {"categorical","categorical"};
    size_t seed = 10;


    baxcat::SyntheticDataGenerator sdg(num_rows, view_weights, category_weights,
        category_separation, datatypes, seed);

    auto data = sdg.getData();

    BOOST_REQUIRE_EQUAL( data.size(), 2 );
    BOOST_REQUIRE_EQUAL( data[0].size(), num_rows );
    BOOST_REQUIRE_EQUAL( data[1].size(), num_rows );
}
// TODO: value checks
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_SUITE_END ()
