
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
#include <memory>

#include "container.hpp"
#include "prng.hpp"
#include "feature.hpp"
#include "helpers/feature_tree.hpp"
#include "datatypes/continuous.hpp"

// What a feature tree does:
// - Insert/remove/return elements with unique indices
// What a feature tree doesn't do:
// - Insert/remove/return elements with non-unique indices
// - remove/return elements that don't exist


BOOST_AUTO_TEST_SUITE (feature_tree_test)

// Using NormalModel Features to test
using std::vector;
using baxcat::Feature;
using baxcat::BaseFeature;
using baxcat::datatypes::Continuous;

struct Setup{
    Setup(baxcat::PRNG *rng){
        baxcat::DataContainer<double> X1({-2,-1,0,1,2});
        f1 = std::shared_ptr<BaseFeature>(new Feature<Continuous, double>(0,X1,{},rng));
        baxcat::DataContainer<double> X2({-5,-3,0,3,5});
        f2 = std::shared_ptr<BaseFeature>(new Feature<Continuous, double>(1,X2,{},rng));
        baxcat::DataContainer<double> X3({-7,-4,0,4,7});
        f3 = std::shared_ptr<BaseFeature>(new Feature<Continuous, double>(2,X3,{},rng));
    }

    std::shared_ptr<BaseFeature> f1;
    std::shared_ptr<BaseFeature> f2;
    std::shared_ptr<BaseFeature> f3;

};


BOOST_AUTO_TEST_CASE(should_insert_elements){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    Setup s(rng);

    baxcat::helpers::FeatureTree ft;
    BOOST_REQUIRE( ft.empty() );

    ft.insert(s.f1);
    BOOST_REQUIRE( !ft.empty() );
    BOOST_REQUIRE_EQUAL( ft.size(), 1 );

    ft.insert(s.f2);
    BOOST_REQUIRE_EQUAL( ft.size(), 2 );
    ft.insert(s.f3);
    BOOST_REQUIRE_EQUAL( ft.size(), 3 );
}

BOOST_AUTO_TEST_CASE(should_insert_elements_in_order){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    Setup s(rng);

    baxcat::helpers::FeatureTree ft;
    BOOST_REQUIRE( ft.empty() );

    ft.insert(s.f1);
    ft.insert(s.f2);
    ft.insert(s.f3);

    // I totes overloaded begin() and end()
    size_t i = 0;
    for( std::shared_ptr<BaseFeature> f: ft){
        BOOST_CHECK_EQUAL(f.get()->getIndex(), i);
        i++;
    }

    for( size_t f = 0; f < ft.size(); f++)
        BOOST_CHECK_EQUAL(ft.at(f).get()->getIndex(), f);

    BOOST_CHECK_EQUAL(ft.at(0).get()->getIndex(), 0);
    BOOST_CHECK_EQUAL(ft.at(1).get()->getIndex(), 1);
    BOOST_CHECK_EQUAL(ft.at(2).get()->getIndex(), 2);
}

BOOST_AUTO_TEST_CASE(should_return_elements){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    Setup s(rng);

    baxcat::helpers::FeatureTree ft;
    BOOST_REQUIRE( ft.empty() );

    ft.insert(s.f1);
    ft.insert(s.f2);
    ft.insert(s.f3);

    BOOST_REQUIRE_EQUAL( ft.size(), 3 );

    std::shared_ptr<BaseFeature> fr2 = ft[1];
    BOOST_CHECK_EQUAL(fr2.get()->getIndex(), 1);

    std::shared_ptr<BaseFeature> fr1 = ft[0];
    BOOST_CHECK_EQUAL(fr1.get()->getIndex(), 0);

    std::shared_ptr<BaseFeature> fr3 = ft[2];
    BOOST_CHECK_EQUAL(fr3.get()->getIndex(), 2);
}

BOOST_AUTO_TEST_CASE(should_remove_elements){
    static baxcat::PRNG *rng = new baxcat::PRNG(10);
    Setup s(rng);

    baxcat::helpers::FeatureTree ft;
    BOOST_REQUIRE( ft.empty() );

    ft.insert(s.f2);
    ft.insert(s.f1);
    ft.insert(s.f3);

    BOOST_REQUIRE_EQUAL( ft.size(), 3 );
    ft.remove(1);

    BOOST_CHECK_EQUAL( ft.size(), 2 );
    std::shared_ptr<BaseFeature> fr1 = ft.at(0);
    BOOST_CHECK_EQUAL(fr1.get()->getIndex(), 0);
    std::shared_ptr<BaseFeature> fr3 = ft.at(1);
    BOOST_CHECK_EQUAL(fr3.get()->getIndex(), 2);

    ft.remove(0);
    ft.remove(2);

    BOOST_CHECK_EQUAL( ft.size(), 0 );
    BOOST_CHECK( ft.empty() );

}
BOOST_AUTO_TEST_SUITE_END()
