
#include "view.hpp"

using std::vector;
using std::function;
using std::shared_ptr;

namespace baxcat{


View::View(vector< shared_ptr<BaseFeature> > &feature_vec, PRNG *rng)
    : _rng(rng)
{
    _num_rows = feature_vec[0].get()->getN();

    // alpha
    _crp_alpha = _rng->invgamrand(1, 1);

    // construct Z, K, Nk from the prior
    _rng->crpGen(_crp_alpha, _num_rows, _row_assignment, _num_clusters,
                 _cluster_counts);

    // build features tree (insert and reassign)
    for(auto &f: feature_vec){
        _features.insert(f);
        f.get()->reassign(_row_assignment);
    }

    // ASSERT_EQUAL(std::cout, checkPartitions(), 1);
}


View::View(vector< shared_ptr<BaseFeature> > &feature_vec, PRNG *rng,
           double crp_alpha, vector<size_t> row_assignment, bool gibbs_init)
    : _rng(rng), _row_assignment(row_assignment)
{
    _num_rows = feature_vec[0].get()->getN();

    // alpha is a semi-optional argument. If it is less than zero, we'll choose
    // ourself
    _crp_alpha = crp_alpha;
    if (crp_alpha <= 0)
        _crp_alpha = _rng->invgamrand(1, 1);

    // build features tree (insert and reassign)
    for(auto &f: feature_vec)
        _features.insert(f);

    if(gibbs_init){
        this->__gibbsInit();
    }else if(_row_assignment.empty()){
        _rng->crpGen(_crp_alpha, _num_rows, _row_assignment, _num_clusters,
                     _cluster_counts);
    }else{
        ASSERT_EQUAL(std::cout, _row_assignment.size(), _num_rows);
        // build partitions
        _num_clusters = utils::vector_max(_row_assignment)+1;
        _cluster_counts.resize(_num_clusters, 0);
        for(size_t &z : _row_assignment)
            _cluster_counts[z]++;
    }

    // build features tree (insert and reassign)
    if(not gibbs_init){
        for(auto &f: _features){
            f.get()->reassign(_row_assignment);    
        }
    }
    
    // ASSERT_EQUAL(std::cout, checkPartitions(), 1);
}


// init with sequential Gibbs
void View::__gibbsInit()
{
    vector<size_t> rows(_num_rows, 0);
    for(size_t r = 0; r < _num_rows; r++)
        rows[r] = r;

    rows = _rng->shuffle(rows);

    for(auto &f: _features){
        f.get()->clear();
        f.get()->insertElementToSingleton(rows[0]);
    }

    _row_assignment.assign(_num_rows,0);
    // _row_assignment[rows[0]] = 0;
    _cluster_counts = {1};
    _num_clusters = 1;

    double log_alpha = log(_crp_alpha);

    for(size_t i = 1; i < _num_rows; ++i){
        vector<double> logps(_num_clusters+1);
        size_t row = rows[i];
        for(size_t k = 0; k < _num_clusters; ++k)
            logps[k] = rowLogp(row, k, true)+log(static_cast<double>(_cluster_counts[k]));
        
        logps.back() = rowSingletonLogp(row) + log_alpha;

        auto assignment = _rng->lpflip(logps);
        _row_assignment[row] = assignment;

        if(assignment == _num_clusters){
            _cluster_counts.push_back(1);
            ++_num_clusters;
            for(auto &f : _features)
                f.get()->insertElementToSingleton(row);
        }else{
            ++_cluster_counts[assignment];
            for(auto &f : _features)
                f.get()->insertElement(row, assignment);
        }
    }

    ASSERT_EQUAL(std::cout, checkPartitions(), 1);
}


// crp alpha
void View::transitionCRPAlpha()
{
    double n = _num_rows;

    size_t burn = 50;

    // construct crp alpha posterior
    const auto cts = _cluster_counts;
    function<double(double)> loglike = [cts, n](double x){
        return numerics::lcrp(cts, n, x);
    };

    function<double()> draw = [this, n](){
        return _rng->invgamrand(1, 1);
    };

    _crp_alpha = samplers::priormh(loglike, draw, burn, _rng);
}


// row transitions
// ````````````````````````````````````````````````````````````````````````````````````````````````
void View::transitionRows()
{
    // TODO: Parallel split-merge sampling
    vector<size_t> rows(_num_rows, 0);
    for(size_t r = 0; r < _num_rows; r++)
        rows[r] = r;

    rows = _rng->shuffle(rows);

    for(auto row: rows)
        transitionRow(row);

    ASSERT(std::cout, checkPartitions()==1);
}


void View::transitionRow(size_t row, bool assign_to_max_p_cluster)
{
    // double log_crp_denom = log(double(_num_rows-1) + _crp_alpha);
    double log_alpha = log(_crp_alpha);

    size_t assign_start = _row_assignment[row];
    bool is_singleton = (_cluster_counts[assign_start] == 1);

    // vector of probabilities
    vector<double> logps;
    if(is_singleton){
        logps.resize(_num_clusters, 0);
    }else{
        logps.resize(_num_clusters+1, 0);
    }

    // TODO: add m argument for extra auxiliary  parameters
    // get the probability of this row under each category
    for(size_t k = 0; k < _num_clusters; k++){
        double lp = rowLogp(row, k);
        double log_crp_numer;
        if(k == assign_start){
            log_crp_numer = is_singleton ? log_alpha : log(double(_cluster_counts[k])-1.0);
        }else{
            log_crp_numer = log(double(_cluster_counts[k]));
        }

        lp += log_crp_numer;
        logps[k] = lp;
    }
    // if it's not already in a singleton, we need to propose one
    if(!is_singleton){
        double lp = rowSingletonLogp(row) + log_alpha;
        logps.back() = lp;
    }

    // get the new index
    size_t assign_new;
    if(assign_to_max_p_cluster){
        assign_new = utils::argmax(logps);
    }else{
        assign_new = _rng->lpflip(logps);
    }

    // move the row if we need to
    if(assign_start != assign_new){
        if( is_singleton ){
            __destroySingletonCluster(row, assign_start, assign_new);
        } else if (assign_new == _num_clusters) {
            __createSingletonCluster(row, assign_start);
        } else{
            __moveRowToCluster(row, assign_start, assign_new);
        }
    }

    ASSERT(std::cout, checkPartitions()==1);
}


// probabilities
// ````````````````````````````````````````````````````````````````````````````````````````````````
double View::rowLogp(size_t row, size_t query_cluster, bool is_init)
{
    auto current_cluster = _row_assignment[row];
    double lp = 0;

    if(query_cluster == current_cluster and not is_init){
        for(auto &f: _features){
            f.get()->removeElement(row, query_cluster);
            lp += f.get()->elementLogp(row, query_cluster);
            f.get()->insertElement(row, query_cluster);
        }
    }else{
        for(auto &f: _features)
            lp += f.get()->elementLogp(row, query_cluster);
    }
    return lp;
}


double View::rowSingletonLogp(size_t row)
{
    double lp = 0;
    for(auto &f: _features)
        lp += f.get()->singletonLogp(row);

    return lp;
}


double View::logScore()
{
    double log_score = 0; 

    log_score += numerics::lcrp(_cluster_counts, _num_rows, _crp_alpha);
    log_score += dist::inverse_gamma::logPdf(_crp_alpha, 1., 1.);

    for(auto &f: _features)
        log_score += f.get()->logScore();

    return log_score;
}


// cleanup
// ````````````````````````````````````````````````````````````````````````````````````````````````
void View::__destroySingletonCluster(size_t row, size_t to_destroy, size_t move_to)
{
    _row_assignment[row] = move_to;
    for(size_t i = 0; i < _num_rows; ++i)
        if( _row_assignment[i] > to_destroy) // maintain partition order
            _row_assignment[i]--;

    for(auto &f: _features)
        f.get()->destroySingletonCluster(row, to_destroy, move_to);

    _cluster_counts[move_to]++;
    _cluster_counts.erase(_cluster_counts.begin()+to_destroy);
    _num_clusters--;
}

void View::__createSingletonCluster(size_t row, size_t current)
{
    _row_assignment[row] = _num_clusters;
    _num_clusters++;
    _cluster_counts[current]--;
    _cluster_counts.push_back(1);
    for(auto &f: _features)
        f.get()->createSingletonCluster(row, current);
}


void View::__moveRowToCluster(size_t row, size_t move_from, size_t move_to)
{
    _row_assignment[row] = move_to;
    _cluster_counts[move_from]--;
    _cluster_counts[move_to]++;
    for(auto &f: _features)
        f.get()->moveToCluster(row, move_from, move_to);
}


// adding and removing dims
// ````````````````````````````````````````````````````````````````````````````````````````````````
void View::assimilateFeature(std::shared_ptr<BaseFeature> &feature)
{
    feature.get()->reassign(_row_assignment);
    _features.insert( feature );
}


void View::releaseFeature(size_t feature_index)
{
    _features.remove(feature_index);
}


// append
// ````````````````````````````````````````````````````````````````````````````````````````````````
void View::appendRow(vector<double> data, vector<size_t> indices, bool assign_to_max_p_cluster)
{
    ++_num_rows;
    _row_assignment.push_back(_num_clusters);
    ++_num_clusters;
    _cluster_counts.push_back(1);
    for(size_t i=0; i < indices.size(); ++i){
        auto feature_index = indices[i];
        auto datum = data[i];
        _features[feature_index].get()->appendRow(datum);
    }
    transitionRow(_num_rows-1, assign_to_max_p_cluster);
}


void View::popRow()
{
    auto assignment = _row_assignment[_num_rows-1];
    auto counts = _cluster_counts[assignment];
    if(counts == 1){
        //reduce number of rows and clusters
        --_num_rows;
        --_num_clusters;
        // remove from counts
        _cluster_counts.erase(_cluster_counts.begin()+assignment);
        // fix assignment indices
        _row_assignment.pop_back();
        for(auto &a : _row_assignment){
            if(a > assignment)
                --a;
        }
    }else{
        --_cluster_counts[assignment];
        --_num_rows;
        _row_assignment.pop_back();
    }

    for(auto &f : _features)
        f.get()->popRow(assignment);

    ASSERT(std::cout, checkPartitions()==1);
}


// setters
// ````````````````````````````````````````````````````````````````````````````````````````````````
void View::setRowAssignment(std::vector<size_t> new_row_assignment)
{
    _row_assignment = new_row_assignment;
    _num_clusters = utils::vector_max(_row_assignment)+1;
    _cluster_counts.resize(_num_clusters, 0);

    for(size_t &z : _row_assignment)
        ++_cluster_counts[z];

    for(auto &f : _features)
        f.get()->reassign(_row_assignment);
}


// getters
// ````````````````````````````````````````````````````````````````````````````````````````````````
size_t View::getNumFeatures() const
{
    return _features.size();
}


size_t View::getNumRows() const
{
    return _num_rows;
}


size_t View::getNumCategories() const
{
    return _num_clusters;
}


double View::getCRPAlpha() const
{
    return _crp_alpha;
}


std::vector<size_t> View::getRowAssignments() const
{
    ASSERT_EQUAL(std::cout, _row_assignment.size(), _num_rows);
    return _row_assignment;
}


std::vector<size_t> View::getClusterCounts() const
{
    return _cluster_counts;
}


size_t View::getAssignmentOfRow(size_t row) const
{
    return _row_assignment[row];
}


std::vector<size_t> View::getFeatureIndices()
{
    std::vector<size_t> indices;
    for( auto &f : _features)
        indices.push_back(f.get()->getIndex());

    return indices;
}


// debugging
// ````````````````````````````````````````````````````````````````````````````````````````````````
int View::checkPartitions()
{
    if (utils::sum(_cluster_counts) != _num_rows)
        return -2;

    if (_cluster_counts.size() != _num_clusters)
        return -1;

    for(size_t k = 0; k < _cluster_counts.size(); ++k){
        size_t sum_k = 0;
        for(auto z : _row_assignment)
            sum_k += (z==k) ? 1 : 0;
        ASSERT_EQUAL(std::cout, sum_k, _cluster_counts[k]);
    }

    // for(auto f : _features){
    //     size_t Nf = 0;
    //     vector< map<string, double> > suffstats = f.get()->getModelSuffstats();
    //     ASSERT_EQUAL(std::cout, _num_clusters, suffstats.size());
    //     for(auto &s: suffstats)
    //         Nf += size_t(s["n"]+.5);

    //     ASSERT_EQUAL(std::cout, Nf, _num_rows);
    //     if(Nf != _num_rows)
    //         return 0;
    // }
    return 1;
}

} // end namespace
