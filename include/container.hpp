
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

#ifndef baxcat_cxx_container_guard
#define baxcat_cxx_container_guard

#include <vector>

namespace baxcat{

// datacontainer is undefined in general (throw compile-time error)
template <typename T>
class __BCDCAllowed__;

// allow only these types
template <> class __BCDCAllowed__<bool>{};
template <> class __BCDCAllowed__<double>{};
template <> class __BCDCAllowed__<size_t>{};
template <> class __BCDCAllowed__<uint_fast8_t>{};
template <> class __BCDCAllowed__<uint_fast16_t>{};
template <> class __BCDCAllowed__<uint_fast32_t>{};


// base template for integral types
template <typename T>
class DataContainer : __BCDCAllowed__<T>
{
    std::vector<T> _data;
    std::vector<bool> _is_initalized;
public:
	DataContainer(){};

    DataContainer(size_t N)
    {
        _data.resize(N);
        _is_initalized.resize(N);
    };

    DataContainer(std::vector<double> data)
    {
        _data.resize(data.size());
        _is_initalized.resize(data.size());
        for(size_t i = 0; i < data.size(); ++i){
            double x_double = data[i]+.5; // add .5 then truncate (avoid cast to lower value)
            if( !std::isnan(x_double ) ){
                _data[i] = (T)x_double;
                _is_initalized[i] = true;
            }
        }
    }

    void set(size_t index, T value)
    {
        _data[index] = value;
        _is_initalized[index] = true;
    }

    void append(T value)
    {
        _data.push_back( value );
        _is_initalized.push_back(true);
    }

    void cast_and_append(double value)
    {
        _data.push_back( static_cast<T>(value+.5) );
        _is_initalized.push_back(true);
    }

    void append_unset_element()
    {
        _data.push_back(0);
        _is_initalized.push_back(false);
    }

    void pop_back()
    {
        _is_initalized.pop_back();
        _data.pop_back();
    }

    void unset(size_t index)
    {
        _is_initalized[index] = false;
    }

    bool is_set(size_t index) const
    {
        return _is_initalized[index];
    }

    bool is_missing(size_t index) const
    {
        return !_is_initalized[index];
    }

    T at(size_t index) const
    {
    	return _data[index];
    }

    size_t size() const
    {
        return _data.size();
    }

    std::vector<double> getSetData() const
    {
        std::vector<double> set_data;
        for(size_t i = 0; i < _data.size(); ++i){
            if(_is_initalized[i])
                set_data.push_back(static_cast<double>(_data[i]));
        }
        return set_data;
    }

    void load_and_cast_data(std::vector<double> data)
    {
        _data.resize(data.size());
        _is_initalized.resize(data.size());
        for(size_t i = 0; i < data.size(); ++i){
            double x_double = data[i]+.5;
            if( !std::isnan(x_double ) ){
                _data[i] = (T)x_double;
                _is_initalized[i] = true;
            }
        }
    }

};


// partial specialization for doubles
//`````````````````````````````````````````````````````````````````````````````````````````````````
// TODO: move semantics for?
template<>
inline DataContainer<double>::DataContainer(std::vector<double> data)
{
    _is_initalized.resize(data.size());
    _data = data;
    for(size_t i = 0; i < data.size(); ++i){
        if(!std::isnan( _data[i]))
            _is_initalized[i] = true;
    }
};


template<>
inline void DataContainer<double>::load_and_cast_data(std::vector<double> data)
{
    _is_initalized.resize(data.size());
    _data = data;
    for(size_t i = 0; i < data.size(); ++i){
        if(!std::isnan(_data[i]))
            _is_initalized[i] = true;
    }
};


template<>
inline void DataContainer<double>::cast_and_append(double value)
{
    _is_initalized.push_back(true);
    _data.push_back( value );
};


template<>
inline std::vector<double> DataContainer<double>::getSetData() const
{
    std::vector<double> set_data;
    for(size_t i = 0; i < _data.size(); ++i){
        if(_is_initalized[i])
            set_data.push_back(_data[i]);
    }
    return set_data;
};


// partial specialization for bools
//`````````````````````````````````````````````````````````````````````````````````````````````````
template<>
inline DataContainer<bool>::DataContainer(std::vector<double> data)
{
    _is_initalized.resize(data.size());
    _data.resize(data.size());
    for(size_t i = 0; i < data.size(); ++i){
        if(!std::isnan(data[i])){
            _is_initalized[i] = true;
            _data[i] = static_cast<bool>(data[i]);
        }
    }
};


template<>
inline void DataContainer<bool>::load_and_cast_data(std::vector<double> data)
{
    _is_initalized.resize(data.size());
    _data.resize(data.size());
    for(size_t i = 0; i < data.size(); ++i){
        if(!std::isnan( data[i])){
            _is_initalized[i] = true;
            _data[i] = static_cast<bool>(data[i]);
        }
    }
};


template<>
inline void DataContainer<bool>::cast_and_append(double value)
{
    _is_initalized.push_back(true);
    _data.push_back(static_cast<bool>(value));
};


template<>
inline std::vector<double> DataContainer<bool>::getSetData() const
{
    std::vector<double> set_data;
    for(size_t i = 0; i < _data.size(); ++i){
        if(_is_initalized[i])
            set_data.push_back(_data[i] ? 1.0 : 0.0);
    }
    return set_data;
};


} // end namespace baxcat

#endif
