
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

#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include "feature.hpp"
#include "numerics.hpp"
#include "utils.hpp"
#include "test_utils.hpp"
#include "datatypes/continuous.hpp"
#include "prng.hpp"


BOOST_AUTO_TEST_SUITE (test_feature)

// I'm using Continuous to test feature
using std::vector;
using std::map;
using std::string;
using baxcat::datatypes::Continuous;

const double EPSILON = 10E-10;

baxcat::Feature<Continuous, double> Setup(baxcat::PRNG *rng)
{
    unsigned int index = 0;

    baxcat::DataContainer<double> data({1,2,3,4,5});
    vector<size_t> assignment = {0,0,0,0,0};

    baxcat::Feature<Continuous, double> feature(index, data, {}, assignment, rng);

    return feature;
}

//  Constructor test
// ````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(constructor_should_produce_valid_object){
   unsigned int index = 0;

   baxcat::DataContainer<double> data({1,2,3,4,5});
   vector<size_t> assignment = {0,0,0,0,0};

   static baxcat::PRNG *rng = new baxcat::PRNG(10);

   baxcat::Feature<Continuous, double> feature(index, data, {}, assignment, rng);

   BOOST_REQUIRE( feature.getIndex() == 0 );

   // check suffstats
   vector<map<string, double>> suffstats = feature.getModelSuffstats();
   BOOST_REQUIRE( suffstats.size() == 1);
   BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["n"], 5, EPSILON );
   BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x"], 15, EPSILON );
   BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x_sq"], 55, EPSILON );

    // check hypers
   vector<map<string, double>> hypers = feature.getModelHypers();
   BOOST_REQUIRE( hypers.size() == 1);
   BOOST_REQUIRE( hypers[0].size() == 4 );
   BOOST_REQUIRE( feature.getN() == 5 );
}


//  insert/remove test
// ````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(remove_all_should_clear_suffstats){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    auto feature = Setup(rng);

    feature.removeElement(0,0);
    feature.removeElement(1,0);
    feature.removeElement(2,0);
    feature.removeElement(3,0);
    feature.removeElement(4,0);

    vector<map<string, double>> suffstats = feature.getModelSuffstats();
    BOOST_REQUIRE( suffstats.size() == 1);
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["n"], 0, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x"], 0, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x_sq"], 0, EPSILON );
}

BOOST_AUTO_TEST_CASE(add_should_increment_suffstats){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    auto feature = Setup(rng);

    feature.insertElement(1,0);

    vector<map<string, double>> suffstats = feature.getModelSuffstats();

    BOOST_REQUIRE( suffstats.size() == 1);
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["n"], 6, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x"], 15 + 2, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x_sq"], 55 + 2*2, EPSILON );

    feature.insertElement(2,0);

    suffstats = feature.getModelSuffstats();

    BOOST_REQUIRE( suffstats.size() == 1);
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["n"], 7, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x"], 15 + 2 + 3, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x_sq"], 55 + 2*2 + 3*3, EPSILON );
}

//  Reassign
// ````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(reassign_should_create_clusters){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    auto feature = Setup(rng);

    vector<size_t> assignment = {0,1,2,3,4};

    feature.reassign(assignment);

    vector<map<string, double>> suffstats = feature.getModelSuffstats();
    vector<map<string, double>> hypers = feature.getModelHypers();

    BOOST_REQUIRE(suffstats.size() == 5);
    BOOST_REQUIRE(hypers.size() == 5);
}

BOOST_AUTO_TEST_CASE(reassign_set_cluster_hypers){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    auto feature = Setup(rng);

    vector<size_t> assignment = {0,1,2,3,4};

    feature.reassign(assignment);

    vector<map<string, double>> hypers = feature.getModelHypers();

    BOOST_REQUIRE( hypers.size() == 5);

    BOOST_CHECK( hypers[0]["m"] == hypers[4]["m"] );
    BOOST_CHECK( hypers[0]["r"] == hypers[4]["r"] );
    BOOST_CHECK( hypers[0]["s"] == hypers[4]["s"] );
    BOOST_CHECK( hypers[0]["nu"] == hypers[4]["nu"] );
}

BOOST_AUTO_TEST_CASE(reassign_set_cluster_suffstats){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    auto feature = Setup(rng);

    vector<size_t> assignment = {0,1,2,3,4};

    feature.reassign(assignment);

    vector<map<string, double>> suffstats = feature.getModelSuffstats();

    BOOST_REQUIRE( suffstats.size() == 5);

    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x"], 1, EPSILON);
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x_sq"], 1*1, EPSILON);

    BOOST_CHECK_CLOSE_FRACTION( suffstats[1]["sum_x"], 2, EPSILON);
    BOOST_CHECK_CLOSE_FRACTION( suffstats[1]["sum_x_sq"], 2*2, EPSILON);

    BOOST_CHECK_CLOSE_FRACTION( suffstats[2]["sum_x"], 3, EPSILON);
    BOOST_CHECK_CLOSE_FRACTION( suffstats[2]["sum_x_sq"], 3*3, EPSILON);

    BOOST_CHECK_CLOSE_FRACTION( suffstats[3]["sum_x"], 4, EPSILON);
    BOOST_CHECK_CLOSE_FRACTION( suffstats[3]["sum_x_sq"], 4*4, EPSILON);

    BOOST_CHECK_CLOSE_FRACTION( suffstats[4]["sum_x"], 5, EPSILON);
    BOOST_CHECK_CLOSE_FRACTION( suffstats[4]["sum_x_sq"], 5*5, EPSILON);
}

//  Cleanup and element-move methods
// ````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(create_singleton_cluster_should_create_new_cluster)
{
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    auto feature = Setup(rng);

    BOOST_REQUIRE(feature.getNumClusters() == 1);

    feature.createSingletonCluster( 1, 0 );

    BOOST_REQUIRE(feature.getNumClusters() == 2);
}

BOOST_AUTO_TEST_CASE(create_singleton_cluster_should_update_suffstats_and_hypers)
{
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    auto feature = Setup(rng);

    vector<map<string, double>> suffstats = feature.getModelSuffstats();
    BOOST_REQUIRE(suffstats.size() == 1);
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x"], 15, EPSILON);
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x_sq"], 55, EPSILON);

    feature.createSingletonCluster( 1, 0 );

    suffstats = feature.getModelSuffstats();
    BOOST_REQUIRE(suffstats.size() == 2);
    BOOST_CHECK_CLOSE_FRACTION(suffstats[0]["sum_x"], 15-2, EPSILON);
    BOOST_CHECK_CLOSE_FRACTION(suffstats[0]["sum_x_sq"], 55-2*2, EPSILON);
    BOOST_CHECK_CLOSE_FRACTION(suffstats[1]["sum_x"], 2, EPSILON);
    BOOST_CHECK_CLOSE_FRACTION(suffstats[1]["sum_x_sq"], 2*2, EPSILON);

    vector<map<string, double>> hypers = feature.getModelHypers();
    BOOST_CHECK_EQUAL(hypers[0]["m"], hypers[1]["m"]);
    BOOST_CHECK_EQUAL(hypers[0]["r"], hypers[1]["r"]);
    BOOST_CHECK_EQUAL(hypers[0]["s"], hypers[1]["s"]);
    BOOST_CHECK_EQUAL(hypers[0]["nu"], hypers[1]["nu"]);
}

BOOST_AUTO_TEST_CASE(move_should_update_suffstats){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    auto feature = Setup(rng);

    vector<size_t> assignment = {0,0,0,1,1};
    vector<map<string, double>> suffstats;

    feature.reassign(assignment);
    suffstats = feature.getModelSuffstats();

    // check starting suffstats
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x"], 6, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x_sq"], 14, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[1]["sum_x"], 9, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[1]["sum_x_sq"], 41, EPSILON );

    // move X[2]=3 from clusters[0] to clusters[1]
    feature.moveToCluster(2, 0, 1);
    suffstats = feature.getModelSuffstats();
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x"], 6-3, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x_sq"], 14-3*3, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[1]["sum_x"], 9+3, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[1]["sum_x_sq"], 41+3*3, EPSILON );
}

BOOST_AUTO_TEST_CASE(destroy_singleton_should_remove_cluster){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    auto feature = Setup(rng);

    vector<size_t> assignment = {0,0,0,0,1};

    feature.reassign(assignment);;

    auto suffstats = feature.getModelSuffstats();
    BOOST_REQUIRE(suffstats.size() == 2);

    // moves X[4] out of clusters[1] to clusters[0] and destroys clusters[1]
    feature.destroySingletonCluster(4, 1, 0);

    suffstats = feature.getModelSuffstats();
    BOOST_REQUIRE(suffstats.size() == 1);
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x"], 15, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x_sq"], 55, EPSILON );
}

BOOST_AUTO_TEST_CASE(pull_data_continuous_should_pull_correct_data)
{
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    auto feature = Setup(rng);

    auto data_f = feature.getData();
    vector<double> data = {1,2,3,4,5};
    BOOST_REQUIRE( baxcat::test_utils::areIdentical(data,data_f) );
}

BOOST_AUTO_TEST_CASE(geweke_clear_continuous_should_delete_suffstats)
{
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    auto feature = Setup(rng);

    // first check that suffstats are set
    auto suffstats = feature.getModelSuffstats();
    BOOST_REQUIRE( suffstats.size() == 1);
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["n"], 5, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x"], 15, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x_sq"], 55, EPSILON );

    feature.__geweke_clear();

    suffstats = feature.getModelSuffstats();
    BOOST_REQUIRE( suffstats.size() == 1);
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["n"], 0, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x"], 0, EPSILON );
    BOOST_CHECK_CLOSE_FRACTION( suffstats[0]["sum_x_sq"], 0, EPSILON );
}

BOOST_AUTO_TEST_CASE(geweke_clear_continuous_should_unset_data)
{
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    auto feature = Setup(rng);

    // first check that suffstats are set
    auto data = feature.getData();
    BOOST_CHECK( data.size() == 5 );

    feature.__geweke_clear();

    data = feature.getData();
    BOOST_CHECK( data.empty() );
}

BOOST_AUTO_TEST_CASE(geweke_clear_and_resample_should_create_new_data)
{
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    auto feature = Setup(rng);

    // first check that suffstats are set
    auto data_start = feature.getData();
    BOOST_CHECK( data_start.size() == 5 );

    feature.__geweke_clear();

    auto data_cleared = feature.getData();
    BOOST_CHECK( data_cleared.empty() );

    for(size_t i = 0; i < 5; ++i){
        feature.__geweke_resampleRow(i, 0, rng);
        auto data_update = feature.getData();
        BOOST_CHECK( data_update.size() == i+1 );
        BOOST_CHECK( data_update[i] != data_start[i] );
    }
}

// test popRow()
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(pop_row_should_remove_element_non_singleton)
{
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    auto feature = Setup(rng);

    auto data = feature.getData();
    auto n_elem = data.size();
    auto last_element =  data[n_elem-1];

    size_t K_start = feature.getNumClusters();

    feature.popRow(0);
    data = feature.getData();

    BOOST_REQUIRE_EQUAL( data.size(), n_elem-1 );
    BOOST_REQUIRE( data[data.size()-1] != last_element );

    size_t K_end = feature.getNumClusters();

    BOOST_REQUIRE_EQUAL( K_start, K_end );

}

BOOST_AUTO_TEST_CASE(pop_row_should_remove_element_singleton)
{
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    unsigned int index = 0;

    baxcat::DataContainer<double> data({1,2,3,4,5});
    vector<size_t> assignment = {0,0,0,0,1};

    baxcat::Feature<Continuous, double> feature(index, data, {}, assignment, rng);

    auto data_out = feature.getData();
    auto n_elem = data.size();
    
    size_t K_start = feature.getNumClusters();

    // remove element from singleton
    feature.popRow(1);
    data_out = feature.getData();

    size_t K_end = feature.getNumClusters();

    BOOST_REQUIRE_EQUAL( data_out.size(), n_elem-1 );
    BOOST_REQUIRE_EQUAL( K_end, K_start-1 );
    BOOST_CHECK( data_out[data_out.size()-1] == 4);

}

// Clear
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(clear_should_remove_clusters)
{
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    unsigned int index = 0;

    baxcat::DataContainer<double> data({1,2,3,4,5});
    vector<size_t> assignment = {0,0,1,1,2};

    baxcat::Feature<Continuous, double> feature(index, data, {}, assignment, rng);

    size_t K_start = feature.getNumClusters();
    BOOST_REQUIRE_EQUAL(K_start, 3);

    feature.clear();

    size_t K_end = feature.getNumClusters();
    BOOST_REQUIRE_EQUAL(K_end, 0);
}

// Clear
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(insert_to_singleton_should_work_with_set_values)
{
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    unsigned int index = 0;

    baxcat::DataContainer<double> data({1,2,3,4,5});
    vector<size_t> assignment = {0,0,1,1,2};

    baxcat::Feature<Continuous, double> feature(index, data, {}, assignment, rng);

    feature.clear();

    BOOST_REQUIRE_EQUAL(feature.getNumClusters(), 0);

    feature.insertElementToSingleton(0);
    BOOST_REQUIRE_EQUAL(feature.getNumClusters(), 1);

    feature.insertElementToSingleton(1);
    BOOST_REQUIRE_EQUAL(feature.getNumClusters(), 2);

    feature.insertElementToSingleton(2);
    BOOST_REQUIRE_EQUAL(feature.getNumClusters(), 3);

    feature.insertElementToSingleton(3);
    BOOST_REQUIRE_EQUAL(feature.getNumClusters(), 4);

    feature.insertElementToSingleton(4);
    BOOST_REQUIRE_EQUAL(feature.getNumClusters(), 5);
}

BOOST_AUTO_TEST_CASE(insert_to_singleton_should_work_with_unset_values)
{
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    unsigned int index = 0;

    baxcat::DataContainer<double> data({1,NAN,3,NAN,5});
    vector<size_t> assignment = {0,0,1,1,2};

    baxcat::Feature<Continuous, double> feature(index, data, {}, assignment, rng);

    feature.clear();

    BOOST_REQUIRE_EQUAL(feature.getNumClusters(), 0);

    feature.insertElementToSingleton(0);
    BOOST_REQUIRE_EQUAL(feature.getNumClusters(), 1);

    feature.insertElementToSingleton(1);
    BOOST_REQUIRE_EQUAL(feature.getNumClusters(), 2);

    feature.insertElementToSingleton(2);
    BOOST_REQUIRE_EQUAL(feature.getNumClusters(), 3);

    feature.insertElementToSingleton(3);
    BOOST_REQUIRE_EQUAL(feature.getNumClusters(), 4);

    feature.insertElementToSingleton(4);
    BOOST_REQUIRE_EQUAL(feature.getNumClusters(), 5);
}

BOOST_AUTO_TEST_SUITE_END()
