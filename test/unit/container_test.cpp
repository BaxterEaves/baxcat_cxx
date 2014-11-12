
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
#include <cmath>
#include <vector>

#include "container.hpp"

BOOST_AUTO_TEST_SUITE (baxcat_container_test)

// Check specialization code (double vs unisgned integral)
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(should_init_appropriate_data_for_uint){
    std::vector<double> X = {1,2,3,4,5};
    baxcat::DataContainer<unsigned int> Y(X);

    BOOST_REQUIRE_EQUAL( Y.at(0), X[0] );
    BOOST_REQUIRE_EQUAL( Y.at(1), X[1] );
    BOOST_REQUIRE_EQUAL( Y.at(2), X[2] );
    BOOST_REQUIRE_EQUAL( Y.at(3), X[3] );
    BOOST_REQUIRE_EQUAL( Y.at(4), X[4] );
}

BOOST_AUTO_TEST_CASE(should_cast_data_at_init_for_uint){
    std::vector<double> X = {1.1, 2.1, 3.1, 4.1, 5.1};
    baxcat::DataContainer<unsigned int> Y(X);

    BOOST_REQUIRE_EQUAL( Y.at(0), 1 );
    BOOST_REQUIRE_EQUAL( Y.at(1), 2 );
    BOOST_REQUIRE_EQUAL( Y.at(2), 3 );
    BOOST_REQUIRE_EQUAL( Y.at(3), 4 );
    BOOST_REQUIRE_EQUAL( Y.at(4), 5 );

    std::vector<double> X2 = {.9, 1.9, 2.9, 3.9, 4.9};
    baxcat::DataContainer<unsigned int> Y2(X2);

    BOOST_REQUIRE_EQUAL( Y2.at(0), 1 );
    BOOST_REQUIRE_EQUAL( Y2.at(1), 2 );
    BOOST_REQUIRE_EQUAL( Y2.at(2), 3 );
    BOOST_REQUIRE_EQUAL( Y2.at(3), 4 );
    BOOST_REQUIRE_EQUAL( Y2.at(4), 5 );
}

BOOST_AUTO_TEST_CASE(should_load_appropriate_data_for_uint){
    std::vector<double> X = {1,2,3,4,5};
    baxcat::DataContainer<unsigned int> Y;

    Y.load_and_cast_data(X);

    BOOST_REQUIRE_EQUAL( Y.at(0), X[0] );
    BOOST_REQUIRE_EQUAL( Y.at(1), X[1] );
    BOOST_REQUIRE_EQUAL( Y.at(2), X[2] );
    BOOST_REQUIRE_EQUAL( Y.at(3), X[3] );
    BOOST_REQUIRE_EQUAL( Y.at(4), X[4] );
}

BOOST_AUTO_TEST_CASE(should_cast_data_at_load_for_uint){
    std::vector<double> X = {1.1, 2.1, 3.1, 4.1, 5.1};
    baxcat::DataContainer<unsigned int> Y;

    Y.load_and_cast_data(X);

    BOOST_REQUIRE_EQUAL( Y.at(0), 1 );
    BOOST_REQUIRE_EQUAL( Y.at(1), 2 );
    BOOST_REQUIRE_EQUAL( Y.at(2), 3 );
    BOOST_REQUIRE_EQUAL( Y.at(3), 4 );
    BOOST_REQUIRE_EQUAL( Y.at(4), 5 );

    std::vector<double> X2 = {.9, 1.9, 2.9, 3.9, 4.9};
    baxcat::DataContainer<unsigned int> Y2;

    Y2.load_and_cast_data(X2);

    BOOST_REQUIRE_EQUAL( Y2.at(0), 1 );
    BOOST_REQUIRE_EQUAL( Y2.at(1), 2 );
    BOOST_REQUIRE_EQUAL( Y2.at(2), 3 );
    BOOST_REQUIRE_EQUAL( Y2.at(3), 4 );
    BOOST_REQUIRE_EQUAL( Y2.at(4), 5 );
}

// double
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(should_init_appropriate_data_for_double){
    std::vector<double> X = {1,2,3,4,5};
    baxcat::DataContainer<double> Y(X);

    BOOST_REQUIRE_EQUAL( Y.at(0), X[0] );
    BOOST_REQUIRE_EQUAL( Y.at(1), X[1] );
    BOOST_REQUIRE_EQUAL( Y.at(2), X[2] );
    BOOST_REQUIRE_EQUAL( Y.at(3), X[3] );
    BOOST_REQUIRE_EQUAL( Y.at(4), X[4] );
}

BOOST_AUTO_TEST_CASE(should_load_appropriate_data_for_double){
    std::vector<double> X = {1,2,3,4,5};
    baxcat::DataContainer<double> Y;

    Y.load_and_cast_data(X);

    BOOST_REQUIRE_EQUAL( Y.at(0), X[0] );
    BOOST_REQUIRE_EQUAL( Y.at(1), X[1] );
    BOOST_REQUIRE_EQUAL( Y.at(2), X[2] );
    BOOST_REQUIRE_EQUAL( Y.at(3), X[3] );
    BOOST_REQUIRE_EQUAL( Y.at(4), X[4] );
}

// check nan values for double, int, and boolean types
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(nan_should_not_have_unset_values_double){
    std::vector<double> X = {1,NAN,3,NAN,5};
    baxcat::DataContainer<double> Y;

    Y.load_and_cast_data(X);

    BOOST_REQUIRE(not Y.is_missing(0));
    BOOST_REQUIRE(not Y.is_missing(2));
    BOOST_REQUIRE(not Y.is_missing(4));
    BOOST_REQUIRE(Y.is_missing(1));
    BOOST_REQUIRE(Y.is_missing(3));
}

BOOST_AUTO_TEST_CASE(nan_should_not_have_unset_values_size_t){
    std::vector<double> X = {1,NAN,3,NAN,5};
    baxcat::DataContainer<size_t> Y;

    Y.load_and_cast_data(X);

    BOOST_REQUIRE(not Y.is_missing(0));
    BOOST_REQUIRE(not Y.is_missing(2));
    BOOST_REQUIRE(not Y.is_missing(4));
    BOOST_REQUIRE(Y.is_missing(1));
    BOOST_REQUIRE(Y.is_missing(3));
}

BOOST_AUTO_TEST_CASE(nan_should_not_have_unset_values_bool){
    std::vector<double> X = {0,NAN,1,NAN,1};
    baxcat::DataContainer<bool> Y;

    Y.load_and_cast_data(X);

    BOOST_REQUIRE(not Y.is_missing(0));
    BOOST_REQUIRE(not Y.is_missing(2));
    BOOST_REQUIRE(not Y.is_missing(4));
    BOOST_REQUIRE(Y.is_missing(1));
    BOOST_REQUIRE(Y.is_missing(3));
}

// check append
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(append_should_add_set_element_to_back){
    std::vector<double> X = {1,2,3,4,5};
    baxcat::DataContainer<size_t> Y(X);

    BOOST_REQUIRE( Y.size() == 5 );

    Y.append(12);

    BOOST_REQUIRE( Y.size() == 6 );
    BOOST_REQUIRE_EQUAL( Y.at(5), 12 );

    Y.append(133);

    BOOST_REQUIRE( Y.size() == 7 );
    BOOST_REQUIRE_EQUAL( Y.at(6), 133 );
}

BOOST_AUTO_TEST_CASE(append_unset_should_add_unsetset_element_to_back){
    std::vector<double> X = {1,2,3,4,5};
    baxcat::DataContainer<size_t> Y(X);

    BOOST_CHECK( Y.size() == 5 );

    Y.append_unset_element();

    BOOST_CHECK( Y.size() == 6 );

    Y.append_unset_element();

    BOOST_CHECK( Y.size() == 7 );

}

BOOST_AUTO_TEST_CASE(append_unset_should_add_unsetset_element_to_back_bool){
    std::vector<double> X = {true, false, true, false, true};
    baxcat::DataContainer<bool> Y(X);

    BOOST_CHECK( Y.size() == 5 );

    Y.append_unset_element();

    BOOST_CHECK( Y.size() == 6 );

    Y.append_unset_element();

    BOOST_CHECK( Y.size() == 7 );
}

BOOST_AUTO_TEST_CASE(cast_and_append_should_add_element_to_back){
    std::vector<double> X = {1,2,3,4,5};
    baxcat::DataContainer<size_t> Y(X);

    BOOST_CHECK( Y.size() == 5 );

    Y.cast_and_append(5.0001);

    BOOST_CHECK( Y.size() == 6 );
    BOOST_CHECK( Y.at(5) == 5 );

    std::vector<double> X2 = {0,1,0,1,0};
    baxcat::DataContainer<bool> Y2(X2);

    BOOST_CHECK( Y2.size() == 5 );

    Y2.cast_and_append(0.0);

    BOOST_CHECK( Y2.size() == 6 );
    BOOST_CHECK( Y2.at(5) == false );

    Y2.cast_and_append(1.0);

    BOOST_CHECK( Y2.size() == 7 );
    BOOST_CHECK( Y2.at(6) == true );

}
// check pop_back
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(pop_back_should_remove_element){
    std::vector<double> X = {1,2,3,4,5};
    baxcat::DataContainer<size_t> Y(X);

    Y.append(6);

    BOOST_REQUIRE_EQUAL( Y.size(), 6 );
    BOOST_REQUIRE_EQUAL( Y.at(5), 6 );

    Y.pop_back();

    BOOST_REQUIRE_EQUAL( Y.size(), 5 );
    BOOST_REQUIRE_EQUAL( Y.at(4), 5 );
}

// check all type functions
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(set_values_should_change_data_value){
    // init a container with 5 empty slots
    baxcat::DataContainer<unsigned int> Y(5);

    BOOST_CHECK( Y.at(0) != 5);

    Y.set(0, 5);

    BOOST_REQUIRE_EQUAL( Y.at(0), 5);
}

BOOST_AUTO_TEST_CASE(values_should_be_unset_until_set){
    // init a container with 5 empty slots
    baxcat::DataContainer<unsigned int> Y(5);

    BOOST_CHECK( Y.is_missing(0) );
    BOOST_CHECK( Y.is_missing(1) );
    BOOST_CHECK( !Y.is_set(0) );
    BOOST_CHECK( !Y.is_set(1) );

    Y.set(0, 5);

    BOOST_CHECK( !Y.is_missing(0) );
    BOOST_CHECK( Y.is_set(0) );
}

BOOST_AUTO_TEST_CASE(get_set_data_should_only_output_set_values){
    baxcat::DataContainer<double> Y(5);

    Y.set(0,0.0);
    Y.set(2,2.0);
    Y.set(4,4.0);

    auto set_data = Y.getSetData();

    BOOST_REQUIRE_EQUAL( set_data[0], 0.0);
    BOOST_REQUIRE_EQUAL( set_data[1], 2.0);
    BOOST_REQUIRE_EQUAL( set_data[2], 4.0);
}

BOOST_AUTO_TEST_SUITE_END()
