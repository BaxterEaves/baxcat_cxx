
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

#include "helpers/feature_tree.hpp"

using std::vector;
using std::shared_ptr;
// using baxcat::BaseFeature;

namespace baxcat{
namespace helpers {


void FeatureTree::insert(shared_ptr<BaseFeature> feature)
{
    size_t index = feature.get()->getIndex();
    if( _indices.empty() ){
        _indices.push_back(index);
        _features.push_back(feature);
    }else{
        size_t insert_at = baxcat::utils::binary_search(_indices, index);
        _indices.insert( _indices.begin()+insert_at, index );
        _features.insert( _features.begin()+insert_at, feature );
    }
}


void FeatureTree::remove(size_t index)
{
    size_t remove_from = utils::binary_search(_indices, index);
    _indices.erase(_indices.begin()+remove_from);
    _features.erase(_features.begin()+remove_from);
}


shared_ptr<BaseFeature> FeatureTree::operator [](size_t index) const
{
    size_t get_at = baxcat::utils::binary_search(_indices, index);
    return _features[get_at];
}


vector<shared_ptr<BaseFeature>>::iterator FeatureTree::begin()
{
    return _features.begin();
}


vector<shared_ptr<BaseFeature>>::iterator FeatureTree::end()
{
    return _features.end();
}


shared_ptr<BaseFeature> FeatureTree::at(size_t index) const
{
    return _features[index];
}


size_t FeatureTree::size() const
{
    return _features.size();
}


bool FeatureTree::empty() const
{
    return _features.empty();
}

}} // end namsepaces
