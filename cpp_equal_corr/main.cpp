#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <tuple>

using namespace std;
using namespace std::chrono;

// Global random number generator
mt19937 rng;

// Function to create states
vector<vector<vector<int8_t>>> create_states(int length, double gamma, int seed, int num) {
    double p_xx = (-1 + sqrt(gamma)) / (2 * gamma - 2);
    int n10 = static_cast<int>(length * (0.5 - p_xx));

    vector<vector<vector<int8_t>>> state(length, vector<vector<int8_t>>(2, vector<int8_t>(num, 0)));

    vector<int> positions(length);
    iota(positions.begin(), positions.end(), 0);

    for (int n = 0; n < num; ++n) {
        shuffle(positions.begin(), positions.end(), rng);

        for (int i = 0; i < n10; ++i) {
            state[positions[i]][0][n] = 1;
        }
        for (int i = n10; i < 2 * n10; ++i) {
            state[positions[i]][1][n] = 1;
        }
        for (int i = 2 * n10; i < length/2 + n10; ++i) {
            state[positions[i]][0][n] = 1;
            state[positions[i]][1][n] = 1;
        }
    }

    return state;
}

// Function to update states synchronously
void update_state_sync(int del_t, double gamma, vector<vector<vector<int8_t>>>& state) {
    int length = state.size();
    int num = state[0][0].size();

    uniform_int_distribution<int> pos_dist(0, length * 2 - 1);
    uniform_real_distribution<double> random_dist(0.0, 1.0);

    for (int k = 0; k < length * 2 * del_t; ++k) {
        int pos1 = pos_dist(rng);
        int sysindex1 = pos1 / length;
        pos1 = pos1 % length;
        int pos2 = (length + pos1 + 1 - 2 * sysindex1) % length;

        if (random_dist(rng) <= gamma) {
            for (int n = 0; n < num; ++n) {
                if (state[pos1][sysindex1][n] == 1) {
                    if (state[pos2][sysindex1][n] == 0) {
                        state[pos1][sysindex1][n] = 0;
                        state[pos2][sysindex1][n] = 1;
                    }
                }
            }
        }
        else {
            int sysindex2 = (sysindex1 + 1) % 2;
            for (int n = 0; n < num; ++n) {
                if (state[pos1][sysindex1][n] == 1) {
                    if (state[pos2][sysindex1][n] == 0) {
                        if (state[pos1][sysindex2][n] * (1 - state[pos2][sysindex2][n]) == 0) {
                            state[pos1][sysindex1][n] = 0;
                            state[pos2][sysindex1][n] = 1;
                        }
                    }
                }
            }
        }
    }
}

double compute_mean(const vector<double>& vec) {
    return accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
}


tuple<int, int> compute_correlator(const vector<vector<vector<int8_t>>>& state,
                         const vector<vector<vector<int8_t>>>& state_t) {
    int res1 = 0, res2 = 0;
    for (size_t n = 0; n < state[0][0].size(); ++n) {
        for (size_t j = 0; j < state.size(); ++j) {
            res1 += state[j][0][n] * state_t[j][0][n] + state[j][1][n] * state_t[j][1][n];
            res1 += (1-state[j][0][n]) * (1-state_t[j][0][n]) + (1-state[j][1][n]) * (1-state_t[j][1][n]);

            res2 += state[j][0][n] * state_t[j][1][n] + state[j][1][n] * state_t[j][0][n];
            res2 += (1-state[j][0][n]) * (1-state_t[j][1][n]) + (1-state[j][1][n]) * (1-state_t[j][0][n]);
        }
    }
    return make_tuple(res1,res2);
}

int main(int argc, char* argv[]) {
    // Read input arguments
    if (argc < 6) {
        cerr << "Usage: " << argv[0] << " <seed> <num> <gamma> <length> <t_max>" << endl;
        return 1;
    }

    int seed = stoi(argv[1]);
    int num = stoi(argv[2]);
    double gamma = stod(argv[3]);
    int length = stoi(argv[4]);
    int t_max = stoi(argv[5]);

    // Initialize the random number generator with the seed
    rng.seed(seed);

    // Record start time for state creation
    auto start_create = high_resolution_clock::now();
    auto state = create_states(length, gamma, seed, num);
    auto end_create = high_resolution_clock::now();
    duration<double> duration_create = end_create - start_create;

    // Record start time for state update
    auto start_update = high_resolution_clock::now();
    vector<vector<vector<int8_t>>> state_t = state;
    vector<vector<int>> corr(t_max, vector<int>(2, 0.0));

    for (int t = 0; t < t_max; ++t) {
        update_state_sync(1, gamma, state_t);
        auto res = compute_correlator(state, state_t);

        corr[t][0] = get<0>(res) - length * num;
        corr[t][1] = get<1>(res) - length * num;
    }

    auto end_update = high_resolution_clock::now();
    duration<double> duration_update = end_update - start_update;

    // Save runtime information
    ofstream runtime_file("runtime_info_seed_" + to_string(seed) + ".txt");
    runtime_file << fixed << setprecision(6);
    runtime_file << "Time taken by create_states: " << duration_create.count() << " seconds\n";
    runtime_file << "Time taken by update_state_sync: " << duration_update.count() << " seconds\n";

    // Calculate memory usage (rough estimation)
    size_t memory_used = length * 2 * num * sizeof(int8_t);
    runtime_file << "Memory used for storing states: " << memory_used << " bytes\n";
    runtime_file.close();

    // Save correlator data
    ofstream corr_file("corr_seed_" + to_string(seed) + ".txt");
    corr_file << fixed << setprecision(6);
    for (const auto& row : corr) {
        for (const auto& val : row) {
            corr_file << val << ' ';
        }
        corr_file << '\n';
    }
    corr_file.close();

    return 0;
}
