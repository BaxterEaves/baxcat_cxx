
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

#ifndef baxcat_cxx_feature_tree_guard
#define baxcat_cxx_feature_tree_guard

#include <vector>
#include <memory>

#include "utils.hpp"
#include "feature.hpp"

namespace baxcat { namespace helpers {

// Feature tree
// ````````````````````````````````````````````````````````````````````````````
// Hold shared pointers to feature objects. Indexed by the Feature member
// Feature.index. For example for a FeatureTree, ft, ft[1] returns a shared
// pointer to the feature with index 1.
// To acess all features in sequence, use range-based for loops or .at()
class FeatureTree
{
public:
    // Add/remove elements
    // add a feature into the tree
    void insert( std::shared_ptr<baxcat::BaseFeature> f );
    // remove a feature
    void remove( size_t index );

    // access
    // return the pointer to the feature with feature.index index
    std::shared_ptr<baxcat::BaseFeature> operator [](size_t index) const;
    // returns the index'th feature stored
    std::shared_ptr<baxcat::BaseFeature> at(size_t index) const;

    // iterators for range-based for loops
    std::vector<std::shared_ptr<baxcat::BaseFeature>>::iterator begin();
    std::vector<std::shared_ptr<baxcat::BaseFeature>>::iterator end();

    // misc
    // retun the number of elements in the tree
    size_t size() const;
    // returns true if there are no elements in the tree
    bool empty() const;

private:
    std::vector<std::shared_ptr<baxcat::BaseFeature>> _features;
    std::vector<size_t> _indices;
};

}}

#endif
