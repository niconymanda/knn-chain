#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include<vector> 
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <bits/stdc++.h>
using namespace std;
 
vector<int> argsort(vector<double> dists) {
    /*argsort the distances*/
    int n = dists.size();
    vector<pair<double, int>> sorting;
    vector<int> indices;


    for (int i = 0; i < n; ++i) {
        sorting.push_back(make_pair(dists[i], i));
    }

    sort(sorting.begin(), sorting.end());

    for (int i = 0; i < n ; i++) {
        indices.push_back(sorting[i].second);
    }

    return indices;
}

int argmin(vector<double> dists) {
    auto min_it = min_element(dists.begin(), dists.end());
    int min_i = distance(dists.begin(), min_it);
    return min_i;
}


double ward(int size_a, int size_b, vector<double> pos_a, vector<double> pos_b) {
    /*calculates the ward for one cluster to another*/
    int i, n = pos_a.size();
    double result, diff, s;
    result = diff = s = 0.0;

    for (i = 0; i < n; i++) {
        diff = pos_a[i] - pos_b[i];
        result += (diff) * (diff);
    }
    s = static_cast<double>(size_a * size_b) / (size_a + size_b);

    return s * result;
}

vector<int> get_top_k(int i, vector<int> size, vector<vector<double>> pos, unordered_set<int> active, int k) {
    vector<int> active_, sorting, top_k, index;
    vector<double> dists;
    double ds;
    for (auto j = active.begin(); j != active.end(); ++j) {
        if (*j != i) {
            active_.push_back(*j);
            ds = ward( size[i], size[*j], pos[i], pos[*j] );
            dists.push_back( ds );
        }
    }
    sorting = argsort(dists);
    sorting.resize(k);
    for (int index = 0; index < k; index++) {
        top_k.push_back(active_[sorting[index]]);
    }

    return top_k;
}

vector<vector<double>> knn_chain(vector<vector<double>> X, int k = 5) {
    /*Calculates the NN chain algorithm with on the fly distances*/
    // Variable declaration & definition
    int i, j, m, index, m_index, new_index, tmp_size, n = X.size();
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
            std::cout << "2 remaining: " << endl;
            auto it = active.begin();
            int i = *it;
            ++it;
            int j = *it;
            std::cout << i << " " << j << endl;
            tmp_size = size[i] + size[j];
            tmp_dist = sqrt(2 * ward(size[i], size[j], pos[i], pos[j]) );
            dendrogram.push_back({static_cast<double>(i), static_cast<double>(j), tmp_dist, static_cast<double>(tmp_size)});
            return dendrogram;
        }
        // Start new chain
        if (!chain.size()) {
            std::cout << "start new chain";
            i = *active.begin();
            chain.push_back(i);
            tmp_knn = get_top_k(i, size, pos, active, 5);
            knn.push_back(tmp_knn);
        }
        // Continue chain
        while (chain.size()) {
            std::cout << "in chain: i = ";
            i = chain.back();
            std::cout << i << endl;
            tmp_knn = knn.back();
            m = -1;
            std::cout << "1-nn: j = " << tmp_knn[0] << endl;
            for (index = 0; index < tmp_knn.size(); index++) {
                if (active.find(tmp_knn[index]) != active.end()) {
                    m = index;
                    break;
                }
            }
            std::cout << "m = " << m << endl;
            if (m <= 0) {
                if (m < 0) {
                    tmp_knn = get_top_k(i, size, pos, active, 5);
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
                    tmp_dist = ward(size[i], size[*index], pos[i], pos[*index]);
                    dists.push_back(tmp_dist);
                }
                cout << argmin(dists) << endl;
                auto it = next(clusters.begin(), argmin(dists));
                j = *it;
                cout << "j = " << j << endl;
            }
            if (chain.size() > 1 && chain[chain.size()-2] == j) {
                std::cout << "merging " << i << " & " << j << "==> ";
                break;
            }
            chain.push_back(j);
            tmp_knn = get_top_k(j, size, pos, active, 5);
            knn.push_back(tmp_knn);
        }
        // Merging i, j
        tmp_dist = sqrt(2 * ward(size[i], size[j], pos[i], pos[j]) );
        tmp_size = size[i] + size[j];
        dendrogram.push_back({static_cast<double>(i), static_cast<double>(j), tmp_dist, static_cast<double>(tmp_size)});

        // Update Variables
        centroid = {};
        for (index = 0; index < pos[i].size(); index++) {
            centroid.push_back( (size[i] * pos[i][index] + size[j] * pos[j][index] ) / tmp_size );
        }
        pos.push_back(centroid);
        new_index = size.size();
        size.push_back(tmp_size);
        std::cout << new_index << endl;

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
        active.erase(k = i);
        active.erase(k = j);
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
    string filename = "smallX";
    vector<vector<double>> pos;
    ifstream file(filename);
    // Check if the file is open
    if (file.is_open()) {
        string line;
        // Read the file line by line
        while (getline(file, line)) {
            vector<double> row;
            istringstream iss(line);
            double value;
            // Parse the line and store values in the row vector
            while (iss >> value) {
                row.push_back(value);
            }
            // Add the row to the data vector
            pos.push_back(row);
        }
        // Close the file
        file.close();
        std::cout << "managed" << endl;
    } else {
        cerr << "Unable to open file " << filename << std::endl;
        return 1;
    }

    // vector<vector<double>> pos = {{1.0, 2.0}, {4.0, 5.0}, {2.0, 8.0}};
    vector<vector<double>> d = knn_chain(pos);
    for (vector<double> dval : d) {
        for (double val : dval) {
            std::cout << val << " ";
        }
        std::cout << endl;
    }
    // cout << endl;
    // cout << argmin({2,3,0,5,6,0}) << endl;

    return 0;
}