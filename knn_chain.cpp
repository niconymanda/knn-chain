#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <numeric> 
#include <vector> 
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <bits/stdc++.h>
#include <chrono>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using namespace std::chrono;
using namespace std;
namespace py = pybind11;

static inline
int argmin(vector<double>& dists) {
    auto min_it = min_element(dists.begin(), dists.end());
    int min_i = distance(dists.begin(), min_it);
    return min_i;
}

static inline
double ward(int size_a, int size_b, const double *pos_a, const double *pos_b, int dim) {
    /* calculates the ward for one cluster to another */
    double result = 0.0;
    double s = static_cast<double>(size_a * size_b) / (size_a + size_b);
#pragma omp parallel for reduction(+:result)
    for (int i = 0; i < dim; ++i) {
        double diff = pos_a[i] - pos_b[i];
        result += diff * diff;
    }
    return s * result;
}

static inline
vector<int> get_top_k(int i, const vector<int>& size, const vector<vector<double>>& pos, const unordered_set<int>& active, int k, int dim) {
    vector<int> active_, top_k, index;
    vector<double> dists;
    double ds;
    int a = active.size()-1, size_i = size[i];
    const double* pos_i = pos[i].data();
    top_k.reserve(k);
    active_.reserve(a);
    dists.reserve(a);

    for (auto j = active.begin(); j != active.end(); ++j) {
        if (*j != i) {
            active_.push_back(*j);
            ds = ward( size_i, size[*j], pos_i, pos[*j].data(), dim );
            dists.push_back( ds );
        }
    }

    vector<int> indices(a);
    iota(indices.begin(), indices.end(), 0);

    partial_sort(indices.begin(), indices.begin() + k, indices.end(), [&](int a, int b) {
        return dists[a] < dists[b];
    });

    for (int index = 0; index < k; ++index) {
        top_k.push_back(active_[indices[index]]);
    }
    
    for (int index = 0; index < k; ++index) {
        top_k.push_back(active_[indices[index]]);
    }
    return top_k;
}

vector<vector<double>> knn_chain(vector<vector<double>> X, int k = 1) {
    /*Calculates the NN chain algorithm with on the fly distances*/
    // Variable declaration & definition
    int i, j, m, index, new_index, tmp_size, n = X.size(), dim = X[0].size();
    double tmp_dist;
    vector<vector<double>> dendrogram, pos = X;
    vector<int> size, chain, tmp_knn;
    vector<double> dists, centroid;
    vector<vector<int>> knn;
    unordered_set<int> active, tmp_rev_mapping;
    unordered_map<int, int> mapping;
    unordered_map<int, unordered_set<int>> reverse_mapping;
    for (int i = 0; i < n; i++) {
        size.push_back(1);
        mapping[i] = i;
        reverse_mapping[i] = {i};
        active.insert(i);
    }

    while (not active.empty()) {
        // Merge the remaining two clusters
        if (active.size() == 2) {
            auto it = active.begin();
            int i = *it;
            ++it;
            int j = *it;
            tmp_size = size[i] + size[j];
            tmp_dist = sqrt(2 * ward(size[i], size[j], pos[i].data(), pos[j].data(), dim) );
            dendrogram.push_back({static_cast<double>(i), static_cast<double>(j), tmp_dist, static_cast<double>(tmp_size)});
            return dendrogram;
        }
        // Start new chain
        if (!chain.size()) {
            i = *active.begin();
            chain.push_back(i);
            tmp_knn = get_top_k(i, size, pos, active, k, dim);
            knn.push_back(tmp_knn);
        }
        // Continue chain
        while (chain.size()) {
            i = chain.back();
            tmp_knn = knn.back();
            m = -1;
            for (index = 0; index < (int)tmp_knn.size(); index++) {
                if (active.find(tmp_knn[index]) != active.end()) {
                    m = index;
                    break;
                }
            }
            if (m <= 0) {
                if (m < 0) {
                    tmp_knn = get_top_k(i, size, pos, active, k, dim);
                }
                j = tmp_knn[0];
                knn.back() = tmp_knn;
            }
            else {
                unordered_set<int> indices;
                for (index = 0; index < m; index++) {
                    tmp_rev_mapping = reverse_mapping[tmp_knn[index]];
                    indices.insert(tmp_rev_mapping.begin(), tmp_rev_mapping.end());
                }
                unordered_set<int> clusters;
                for (int index : indices) {
                    clusters.insert(mapping[index]);
                }
                clusters.insert(tmp_knn[m]);
                vector<double> dists;
                for (auto index = clusters.begin(); index != clusters.end(); ++index) {
                    tmp_dist = ward(size[i], size[*index], pos[i].data(), pos[*index].data(), dim);
                    dists.push_back(tmp_dist);
                }
                auto it = next(clusters.begin(), argmin(dists));
                j = *it;
            }
            if (chain.size() > 1 && chain[chain.size()-2] == j) {
                break;
            }
            chain.push_back(j);
            tmp_knn = get_top_k(j, size, pos, active, k, dim);
            knn.push_back(tmp_knn);
        }
        // Merging i, j
        tmp_size = size[i] + size[j];
        tmp_dist = sqrt(2 * ward(size[i], size[j], pos[i].data(), pos[j].data(), dim) );
        dendrogram.push_back({static_cast<double>(i), static_cast<double>(j), tmp_dist, static_cast<double>(tmp_size)});

        // Update Variables
        centroid = {};
        for (index = 0; index < (int)pos[i].size(); index++) {
            centroid.push_back( (size[i] * pos[i][index] + size[j] * pos[j][index] ) / tmp_size );
        }
        pos.push_back(centroid);
        new_index = size.size();
        size.push_back(tmp_size);

        // Update Mapping
        unordered_set<int> union_set;
        for (int index : reverse_mapping[i]) {
            union_set.insert(index);
        }
        for (int index : reverse_mapping[j]) {
            union_set.insert(index);
        }
        for (int index : union_set) {
            mapping[index] = new_index;
        }
        reverse_mapping[new_index] = union_set;

        // Update active set
        active.erase(i);
        active.erase(j);
        active.insert(new_index);

        chain.erase(chain.end()-2, chain.end());
        knn.erase(knn.end()-2, knn.end());
    }

    return dendrogram;
}

int main(){
    // TESTING WARD
    // int size_a, size_b;
    // size_a = size_b = 1;
    // vector<double> pos_a = {1.0, 2.0};
    // vector<double> pos_b = {4.0, 5.0};
    // cout << ward(size_a, size_b, pos_a, pos_b);

    // TESTING GET_TOP_K
    // vector<int> size = {1,1,1,1};
    // vector<vector<double>> pos = {{1.0, 2.0}, {4.0, 5.0}, {2.0, 8.0}, {2.0, 10.0}};
    // unordered_set<int> active = {0, 1, 2, 3};
    // vector<int> top_k = get_top_k(0, size, pos, active, 2);

    // TESTING KNN_CHAIN
    string filename = "largeX";
    vector<vector<double>> pos;
    ifstream file(filename);
    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            vector<double> row;
            istringstream iss(line);
            double value;
            while (iss >> value) {
                row.push_back(value);
            }
            pos.push_back(row);
        }
        file.close();
    } else {
        cerr << "Unable to open file " << filename << std::endl;
        return 1;
    }

    // vector<vector<double>> pos = {{1.0, 2.0}, {4.0, 5.0}, {2.0, 8.0}};
    auto start = high_resolution_clock::now();
    vector<vector<double>> d = knn_chain(pos);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << duration.count() * 0.000001 << endl; 
    /*
    Compiling at -O3 ==> 8.53 seconds for X.shape = (10 000, 100)
    => compared to SciPy which executes in 7.568 seconds
    */
    /*
    for (vector<double> dval : d) {
        for (double val : dval) {
            std::cout << val << " ";
        }
        std::cout << endl;
    }
    */
    return 0;
}

PYBIND11_MODULE(knn_chain, m) {
    m.doc() = "knn_chain clustering algorithm";
    m.def("knn_chain", &knn_chain, "knn-chain clustering");
}

