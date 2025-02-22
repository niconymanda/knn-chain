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
#include <immintrin.h> 
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

double ward(double size_a, double size_b, const double* pos_a, const double* pos_b, int dim) {
    /* calculates the ward for one cluster to another */
    __m256d sum = _mm256_setzero_pd();
    double result = 0.0;
    int i;
    double s = static_cast<double>(size_a * size_b) / (size_a + size_b);

    for (i = 0; i + 4 <= dim; i += 4) {
        __m256d a = _mm256_loadu_pd(&pos_a[i]);
		__m256d b = _mm256_loadu_pd(&pos_b[i]);

        // Calculate the difference
        __m256d diff = _mm256_sub_pd(a, b);
        // Square the difference
        __m256d square = _mm256_mul_pd(diff, diff);
        // Accumulate the result
        sum = _mm256_add_pd(sum, square);
    }
    double buffer[4];
    _mm256_storeu_pd(buffer, sum); // Store the SIMD result into an array
    result = buffer[0] + buffer[1] + buffer[2] + buffer[3]; // Sum up the results

    // Handle remaining elems
    for (; i < dim; ++i) {
        double diff = pos_a[i] - pos_b[i];
        result += diff * diff;
    }

    return s * result;
}


void get_top_k(int i, const vector<double>& size, const vector<vector<double>>& pos, const unordered_set<int>& active, int k, int dim, vector<int>* top_k) {
    vector<int> active_;
    vector<double> dists;
    double ds;
    int a = active.size()-1, size_i = size[i];
    const double* pos_i = pos[i].data();
    int p = min(k, a);
    top_k->clear();
    top_k->reserve(p); 
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

    partial_sort(indices.begin(), indices.begin() + p, indices.end(), [&](int a, int b) {
        if (dists[a] == dists[b]) {
            return active_[a] < active_[b]; 
        }
        return dists[a] < dists[b];
    });

    for (int index = 0; index < p; ++index) {
        top_k->push_back(active_[indices[index]]);
    }
}


vector<vector<double>> knn_chain(vector<vector<double>> X, int k = 1, vector<double> weights = {}) {
    /*Calculates the NN chain algorithm with on the fly distances*/
    // Variable declaration & definition
    int i, j, m, index, new_index, tmp_size, n = X.size(), dim = X[0].size();
    double tmp_dist;
    vector<vector<double>> dendrogram, pos = X;
    vector<int> chain, top_k;
    vector<double> size, dists, centroid;
    vector<vector<int>> knn;
    unordered_set<int> active, tmp_rev_mapping;
    unordered_map<int, int> mapping;
    unordered_map<int, unordered_set<int>> reverse_mapping;
    
    pos.reserve(2*n-3);
    top_k.reserve(k);
    dendrogram.reserve(2*n-1);
    centroid.reserve(dim);
    if (weights.empty()) {
        size.reserve(2*n-1);
        size.resize(n, 1.0);
        // size.push_back(1.0);
    }
    else {
        if (weights.size() !=  static_cast<size_t>(n)) {
            throw invalid_argument("The length of the sample weights (" + std::to_string(weights.size()) + ") must be equal to the size of X (" + std::to_string(n) + ").");
        }
        size = weights;
    }

    for (int i = 0; i < n; i++) {
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
            tmp_dist = sqrt( 2 * ward(size[i], size[j], pos[i].data(), pos[j].data(), dim) );
            dendrogram.push_back({static_cast<double>(i), static_cast<double>(j), tmp_dist, static_cast<double>(tmp_size)});
            return dendrogram;
        }
        // Start new chain
        if (!chain.size()) {
            i = *active.begin();
            chain.push_back(i);
            get_top_k(i, size, pos, active, k, dim, &top_k);
            knn.push_back(top_k);
        }
        // Continue chain
        while (chain.size()) {
            i = chain.back();
            top_k = knn.back();
            m = -1;
            for (index = 0; index < (int)top_k.size(); index++) {
                if (active.find(top_k[index]) != active.end()) {
                    m = index;
                    break;
                }
            }
            if (m <= 0) {
                if (m < 0) {
                    get_top_k(i, size, pos, active, k, dim, &top_k);
                }
                j = top_k[0];
                knn.back() = top_k;
            }
            else {
                unordered_set<int> indices;
                for (index = 0; index < m; index++) {
                    tmp_rev_mapping = reverse_mapping[top_k[index]];
                    indices.insert(tmp_rev_mapping.begin(), tmp_rev_mapping.end());
                }
                unordered_set<int> clusters;
                for (int index : indices) {
                    clusters.insert(mapping[index]);
                }
                clusters.insert(top_k[m]);
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
            get_top_k(j, size, pos, active, k, dim, &top_k);
            knn.push_back(top_k);
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

PYBIND11_MODULE(knn_chain, m) {
    m.doc() = "knn_chain clustering algorithm";
    m.def("knn_chain", &knn_chain, "knn-chain clustering", py::arg("X"), py::arg("k") = 1, py::arg("weights") = std::vector<double>{});
}
