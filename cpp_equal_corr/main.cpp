#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <tuple>
#include <unistd.h>
#include <sys/stat.h>

using namespace std;
using namespace std::chrono;

// Global random number generator
mt19937 rng;

// Function to create states
vector<vector<vector<int8_t>>> create_states(int length, double gamma, int num) {
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
        for (int i = 2 * n10; i < length / 2 + n10; ++i) {
            state[positions[i]][0][n] = 1;
            state[positions[i]][1][n] = 1;
        }
    }

    return state;
}

// Function to update states synchronously
void update_state_sync(int del_t, double gamma, vector<vector<vector<int8_t>>> &state) {
    int length = state.size();
    int num = state[0][0].size();

    uniform_int_distribution<int> pos_dist(0, length * 2 - 1);
    uniform_real_distribution<double> random_dist(0.0, 1.0);

    // Generating all needed random values at once
    vector<double> random_values(length * 2 * del_t);
    generate(random_values.begin(), random_values.end(), [&]() { return random_dist(rng); });

    for (int k = 0; k < length * 2 * del_t; ++k) {
        int pos1 = pos_dist(rng);
        int sys_index_1 = pos1 / length;
        pos1 %= length;
        int pos2 = (length + pos1 + 1 - 2 * sys_index_1) % length;

        if (random_values[k] <= gamma) {
            for (int n = 0; n < num; ++n) {
                int8_t &src = state[pos1][sys_index_1][n];
                int8_t &dst = state[pos2][sys_index_1][n];
                if (src * (1 - dst) == 1) {
                    src = 0;
                    dst = 1;
                }
            }
        } else {
            int sys_index_2 = (sys_index_1 + 1) % 2;
            for (int n = 0; n < num; ++n) {
                int8_t &src = state[pos1][sys_index_1][n];
                int8_t &dst = state[pos2][sys_index_1][n];
                if (src * (1 - dst) == 1) {
                    if (state[pos1][sys_index_2][n] * (1 - state[pos2][sys_index_2][n]) == 0) {
                        src = 0;
                        dst = 1;
                    }
                }
            }
        }
    }
}


tuple<int, int> compute_correlation(const vector<vector<vector<int8_t>>> &state,
                                    const vector<vector<vector<int8_t>>> &state_t) {
    int res1 = 0, res2 = 0;
    size_t layers = state.size(), height = state[0].size(), width = state[0][0].size();

    for (size_t j = 0; j < layers; ++j) {
        for (size_t n = 0; n < width; ++n) {
            int state_j0n = state[j][0][n];
            int state_j1n = state[j][1][n];
            int state_t_j0n = state_t[j][0][n];
            int state_t_j1n = state_t[j][1][n];

            res1 += state_j0n * state_t_j0n + state_j1n * state_t_j1n;
            res1 += (1 - state_j0n) * (1 - state_t_j0n) + (1 - state_j1n) * (1 - state_t_j1n);

            res2 += state_j0n * state_t_j1n + state_j1n * state_t_j0n;
            res2 += (1 - state_j0n) * (1 - state_t_j1n) + (1 - state_j1n) * (1 - state_t_j0n);
        }
    }
    return make_tuple(res1, res2);
}


int main(int argc, char *argv[]) {
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
    auto state = create_states(length, gamma, num);
    auto end_create = high_resolution_clock::now();
    duration<double> duration_create = end_create - start_create;

    // Record start time for state update
    auto start_update = high_resolution_clock::now();

    duration<double> duration_update = start_update - start_update;
    duration<double> duration_record = start_update - start_update;

    vector<vector<vector<int8_t>>> state_t = state;
    vector<vector<int>> corr(t_max, vector<int>(2, 0.0));

    for (int t = 0; t < t_max; ++t) {
        start_update = high_resolution_clock::now();
        update_state_sync(1, gamma, state_t);
        duration_update += high_resolution_clock::now() - start_update;

        start_update = high_resolution_clock::now();
        auto res = compute_correlation(state, state_t);
        duration_record += high_resolution_clock::now() - start_update;

        corr[t][0] = get<0>(res) - length * num;
        corr[t][1] = get<1>(res) - length * num;
    }

    duration<double> total_duration = high_resolution_clock::now() - start_create;


    // create directory if it doesn't exist for results
    string sim_base = "gam_" + to_string(int(gamma * 1000))
                      + "_len_" + to_string(length)
                      + "_t_" + to_string(t_max)
                      + "_num_" + to_string(num);

    // Check if the folder already exists
    if (access(sim_base.c_str(), F_OK) != 0) {
        // Folder doesn't exist, create it
        if (mkdir(sim_base.c_str(), 0777) == 0) {
            std::cout << "Folder created successfully." << std::endl;
        } else {
            std::cerr << "Error creating folder." << std::endl;
            return 1;
        }
    } else {
        std::cout << "Folder already exists." << std::endl;
    }



    // Save correlator data
    ofstream corr_file(sim_base + "/" + to_string(seed) + "_" + sim_base + ".txt");
    corr_file << fixed << setprecision(6);
    corr_file << "# Total time: " << total_duration.count() << " seconds\n";
    corr_file << "# Time taken by create_states: " << duration_create.count() << " seconds\n";
    corr_file << "# Time taken by update_state_sync: " << duration_update.count() << " seconds\n";
    corr_file << "# Time taken by recording correlation: " << duration_record.count() << " seconds\n";


    for (const auto &row: corr) {
        for (const auto &val: row) {
            corr_file << val << ' ';
        }
        corr_file << '\n';
    }
    corr_file.close();

    return 0;
}
