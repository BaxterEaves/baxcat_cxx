
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

#ifndef baxcat_cxx_feature_builder_guard
#define baxcat_cxx_feature_builder_guard

#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <map>

#include "container.hpp"
#include "numerics.hpp"
#include "feature.hpp"
#include "debug.hpp"
#include "prng.hpp"
#include "helpers/state_helper.hpp"

// Include every datamodel header here
#include "datatypes/continuous.hpp"
#include "datatypes/categorical.hpp"

using baxcat::datatypes::Continuous;
using baxcat::datatypes::Categorical;

namespace baxcat{
namespace helpers{


// TODO: OPTIMIZATION: construct and store similar data type features in vectors
static std::vector<std::shared_ptr<BaseFeature>> genFeatures(
    std::vector<std::vector<double>> data_in, std::vector<std::string> datatypes,
    std::vector<std::vector<double>> distargs, baxcat::PRNG *rng)
{

    ASSERT_EQUAL(std::cout, data_in.size(), datatypes.size());
    ASSERT_EQUAL(std::cout, data_in.size(), distargs.size());


    std::vector<std::shared_ptr<BaseFeature>> features_out;
    auto converted_datatypes = getDatatypes(datatypes);

    for(size_t i = 0; i < datatypes.size(); ++i){
        if(converted_datatypes[i] == continuous){
            baxcat::DataContainer<double> data(data_in[i]);
            std::shared_ptr<BaseFeature> ptr(new Feature<Continuous, double>(i, data, {}, rng));
            features_out.push_back(ptr);

        }else if(converted_datatypes[i] == categorical){
            // TODO: choose container var type based on counts to reduce RAM requirements
            baxcat::DataContainer<size_t> data(data_in[i]);
            std::shared_ptr<BaseFeature> ptr(new Feature<Categorical, size_t>(i, data, distargs[i], rng));
            features_out.push_back(ptr);

        }else{
            // FIXME: add properr exception
            throw 1;
        }
    }

    return features_out;
}

}} // end namespaces


#endif
