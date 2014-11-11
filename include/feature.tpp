
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

using std::vector;
using std::string;
using std::map;


template<class DataType, typename T>
baxcat::Feature<DataType, T>::Feature(unsigned int idx, baxcat::DataContainer<T> data,
                                      vector<double> args, baxcat::PRNG *rng_ptr)
    : _index(idx), _data(data), _rng(rng_ptr), _N(data.size())
{
    _distargs = args;
    _hyperprior_config = DataType::constructHyperpriorConfig(_data.getSetData());
    _hypers = DataType::initHypers(_hyperprior_config , _rng);
}


// FIXME: figure out how to get delegated constructors to work here
template<class DataType, typename T>
baxcat::Feature<DataType, T>::Feature(unsigned int idx, baxcat::DataContainer<T> data,
                                      vector<double> args, vector<size_t> Z, baxcat::PRNG *rngptr)
    : _index(idx), _data(data), _rng(rngptr), _N(data.size())
{
    _distargs = args;
    _hyperprior_config = DataType::constructHyperpriorConfig(_data.getSetData());
    _hypers = DataType::initHypers(_hyperprior_config , _rng);

    _clusters.clear();
    size_t K = utils::vector_max(Z) + 1;
    for(size_t k = 0; k < K; k++){
        _clusters.emplace_back(_distargs);
        _clusters[k].setHypers(_hypers);
    }

    ASSERT_EQUAL(std::cout, _clusters.size(), K);

    for( size_t i = 0; i < _N; i++)
        this->insertElement( i, Z[i] );
}


// for geweke
template<class DataType, typename T>
baxcat::Feature<DataType, T>::Feature(unsigned int idx, baxcat::DataContainer<T> data,
                                      vector<double> args, baxcat::PRNG *rng_ptr, 
                                      vector<double> hypers, vector<double> hyperprior_config)
    : _index(idx), _data(data), _rng(rng_ptr), _N(data.size()),
      _hyperprior_config(hyperprior_config)
{
    if(hypers.empty()){
        _hypers = DataType::initHypers(_hyperprior_config, _rng);
    }else{
        _hypers = hypers;
    }
    _distargs = args;
}

// Add/remove elements
// ````````````````````````````````````````````````````````````````````````````````````````````````
template<class DataType, typename T>
void baxcat::Feature<DataType, T>::insertElement(size_t row, size_t cluster)
{
    if(_data.is_set(row))
        _clusters[cluster].insertElement(_data.at(row));
}


template<class DataType, typename T>
void baxcat::Feature<DataType, T>::insertElementToSingleton(size_t row)
{
    _clusters.emplace_back(_distargs);
    _clusters.back().setHypers(_hypers);
    this->insertElement(row, _clusters.size()-1);
}


template<class DataType, typename T>
void baxcat::Feature<DataType, T>::removeElement(size_t row, size_t cluster)
{
    if(_data.is_set(row))
        _clusters[cluster].removeElement(_data.at(row));
}


template<class DataType, typename T>
void baxcat::Feature<DataType, T>::insertValue(double value, size_t cluster)
{
    _clusters[cluster].insertElement(T(value));
}


template<class DataType, typename T>
void baxcat::Feature<DataType, T>::removeValue(double value, size_t cluster)
{
    _clusters[cluster].insertElement(T(value));
}


// update hypers
// ````````````````````````````````````````````````````````````````````````````````````````````````
template<class DataType, typename T>
void baxcat::Feature<DataType, T>::updateHypers()
{
    _hypers = DataType::resampleHypers(_clusters, _hyperprior_config, _rng);
    for(auto &cluster : _clusters)
        cluster.setHypers(_hypers);
}


// Probability
// ````````````````````````````````````````````````````````````````````````````````````````````````
template<class DataType, typename T>
double baxcat::Feature<DataType, T>::elementLogp(size_t row, size_t cluster) const
{
    return _data.is_missing(row) ? 0.0 : _clusters[cluster].elementLogp(_data.at(row));
}


template<class DataType, typename T>
double baxcat::Feature<DataType, T>::singletonLogp(size_t row) const
{
    return _data.is_missing(row) ? 0.0 : _clusters[0].singletonLogp(_data.at(row));
}


template<class DataType, typename T>
double baxcat::Feature<DataType, T>::valueLogp(double value, size_t cluster) const
{
    return std::isnan(value) ? 0.0 : _clusters[cluster].elementLogp(T(value));
}


template<class DataType, typename T>
double baxcat::Feature<DataType, T>::singletonValueLogp(double value) const
{
    return std::isnan(value) ? 0.0 : _clusters[0].singletonLogp(T(value));
}


template<class DataType, typename T>
double baxcat::Feature<DataType, T>::clusterLogp(size_t cluster) const
{
    return _clusters[cluster].logp();
}


template<class DataType, typename T>
double baxcat::Feature<DataType, T>::logp() const
{
    double logp = 0;
    // double logp = _clusters[0].hyperpriorLogp(_hyperprior_config);
    for(auto &cluster : _clusters)
        logp += cluster.logp();

    return logp;
}


// Draw
// ````````````````````````````````````````````````````````````````````````````````````````````````
template<class DataType, typename T>
double baxcat::Feature<DataType, T>::drawFromCluster(size_t cluster_idx, baxcat::PRNG *rng)
{
    return double(_clusters[cluster_idx].draw(rng));
}


// Cleanup
// ````````````````````````````````````````````````````````````````````````````````````````````````
template<class DataType, typename T>
void baxcat::Feature<DataType, T>::moveToCluster(size_t row, size_t move_from, size_t move_to)
{
    this->removeElement(row, move_from);
    this->insertElement(row, move_to);
}


template<class DataType, typename T>
void baxcat::Feature<DataType, T>::destroySingletonCluster(size_t row, size_t to_destroy,
                                                           size_t move_to)
{
    this->insertElement(row, move_to);
    _clusters.erase(_clusters.begin() + to_destroy);
}


template<class DataType, typename T>
void baxcat::Feature<DataType, T>::createSingletonCluster(size_t row, size_t current)
{
    this->removeElement(row, current);
    _clusters.emplace_back(_distargs);
    _clusters.back().setHypers(_hypers);
    this->insertElement(row, _clusters.size()-1);
}


template<class DataType, typename T>
void baxcat::Feature<DataType, T>::reassign(std::vector<size_t> assignment)
{
    size_t K_old = _clusters.size();
    size_t K_new = utils::vector_max(assignment) + 1;

    if(K_new < K_old){
        _clusters.resize(K_new, DataType(_distargs));
        for(size_t k = 0; k < K_new; k++)
            _clusters[k].setHypers(_hypers);
    }else{
        _clusters.resize(K_new, DataType(_distargs));
        for(size_t k = 0; k < K_old; k++)
            _clusters[k].clear(_distargs);

        for(size_t k = K_old; k < K_new; k++)
            _clusters[k].setHypers(_hypers);
    }

    ASSERT_EQUAL(std::cout, _clusters.size(), K_new);

    for(size_t i = 0; i < _N; ++i)
        this->insertElement(i, assignment[i]);
}


// Getters
// ````````````````````````````````````````````````````````````````````````````````````````````````
template<class DataType, typename T>
size_t baxcat::Feature<DataType, T>::getIndex() const
{
    return _index;
}


template<class DataType, typename T>
size_t baxcat::Feature<DataType, T>::getN() const
{
    return _N;
}


template<class DataType, typename T>
vector<double> baxcat::Feature<DataType, T>::getHypers() const
{
    return _hypers;
}


template<class DataType, typename T>
map<string, double> baxcat::Feature<DataType, T>::getHypersMap() const
{
    ASSERT(std::cout, _clusters.size() > 0);
    return _clusters[0].getHypersMap();
}


template<class DataType, typename T>
vector<map<string, double>> baxcat::Feature<DataType, T>::getModelHypers() const
{
    ASSERT(std::cout, _clusters.size() > 0);

    vector<map<string, double>> ret;
    for(const DataType &cluster : _clusters)
        ret.push_back(cluster.getHypersMap());

    return ret;
}


template<class DataType, typename T>
vector<map<string, double>> baxcat::Feature<DataType, T>::getModelSuffstats() const
{
    assert(_clusters.size() > 0);

    vector<map<string, double>> ret;
    for(const DataType &cluster : _clusters)
        ret.push_back(cluster.getSuffstatsMap());

    return ret;
}


// TODO: implement so we can use variable return types
template<class DataType, typename T>
std::vector<double> baxcat::Feature<DataType, T>::getData() const
{
    return _data.getSetData();
}


// Setters
// ````````````````````````````````````````````````````````````````````````````````````````````````
template<class DataType, typename T>
void baxcat::Feature<DataType, T>::clear()
{
    _clusters.clear();
}


template<class DataType, typename T>
void baxcat::Feature<DataType, T>::setHypers(map<string, double> hypers_map)
{
    for(auto &cluster : _clusters)
        cluster.setHypersByMap(hypers_map);
    // set feature hypers
    _hypers = _clusters[0].getHypers();
}


template<class DataType, typename T>
void baxcat::Feature<DataType, T>::setHypers(vector<double> hypers_vec)
{
    for(auto &cluster : _clusters)
        cluster.setHypers(hypers_vec);
    // set feature hypers
    _hypers = hypers_vec;
}


template<class DataType, typename T>
void baxcat::Feature<DataType, T>::setHyperConfig(std::vector<double> hyperprior_config)
{
    _hyperprior_config = hyperprior_config;
}


template<class DataType, typename T>
void baxcat::Feature<DataType, T>::appendRow(double datum)
{
    if( std::isnan(datum) ){
        _data.append_unset_element();
    }else{
        _data.cast_and_append(datum);
    }
    _clusters.emplace_back(_distargs);
    _clusters.back().setHypers(_hypers);
    this->insertElement(_data.size()-1, _clusters.size()-1);
}


template<class DataType, typename T>
void baxcat::Feature<DataType, T>::popRow(  size_t cluster_assignment )
{
    --_N;
    auto count = _clusters[cluster_assignment].getCount();
    if(count==1){
        _clusters.erase(_clusters.begin()+cluster_assignment);
    }else{
        if(_data.is_set(_N)){
            auto element = _data.at(_N);
            _clusters[cluster_assignment].removeElement(element);
        }
    }
    _data.pop_back();
}


// Testing
// ````````````````````````````````````````````````````````````````````````````````````````````````
template<class DataType, typename T>
void baxcat::Feature<DataType, T>::__geweke_resampleRow(size_t which_row, size_t which_cluster,
                                                        baxcat::PRNG *rng)
{
    // TODO: When component constant-updating functions are implemented as a part of optimization,
    // we will need to update this code

    // remove data at which_row if it exists
    if(_data.is_set(which_row)){
        T x = _data.at(which_row);
        _clusters[which_cluster].removeElement(x);
    }
    
    T y = _clusters[which_cluster].draw(rng);

    _data.set(which_row, y);
    _clusters[which_cluster].insertElement(y);
}


template<class DataType, typename T>
void baxcat::Feature<DataType, T>::__geweke_clear()
{
    for(size_t i = 0; i < _data.size(); ++i)
        _data.unset(i);

    for(auto &cluster : _clusters)
        cluster.clear(_distargs);
}


template<class DataType, typename T>
void baxcat::Feature<DataType, T>::__geweke_initHypers()
{
    _hypers = DataType::initHypers(_hyperprior_config, _rng);
    for(auto &cluster : _clusters)
        cluster.setHypers(_hypers);
}
