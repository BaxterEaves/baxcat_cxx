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
#include "prng.hpp"
#include "helpers/state_helper.hpp"

// Include every datamodel header here
#include "datatypes/continuous.hpp"

using baxcat::datatypes::Continuous;

namespace baxcat{ namespace helpers{


// OPTIMIZATION: construct and store similar data type features in vectors
static std::vector<std::shared_ptr<BaseFeature>> genFeatures(
    std::vector<std::vector<double>> data_in, std::vector<std::string> datatypes,
    std::vector<std::vector<double>> distargs, baxcat::PRNG *rng)
{
    std::vector<std::shared_ptr<BaseFeature>> features_out;
    auto converted_datatypes = getDatatypes( datatypes );
    assert( data_in.size() == datatypes.size() );

    for( size_t i = 0; i < datatypes.size(); ++i){
        if( converted_datatypes[i] == continuous ){
            baxcat::DataContainer<double> data(data_in[i]);
            std::shared_ptr<BaseFeature> ptr(new Feature<Continuous, double>(i, data, {}, rng));
            features_out.push_back(ptr);
        }else{
            // FIXME: add proper exception
            throw 1;
        }
    }

    return features_out;
}

}} // end namespaces


#endif
