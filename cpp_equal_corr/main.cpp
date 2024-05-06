#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <chrono>
#include <cstring>
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


// function creating the result path recursively
bool createDirectoryRecursively(const std::string &path) {
    size_t pos = 0;
    std::string currentPath;

    // Handle absolute path starting with '/'
    if (path[0] == '/') {
        currentPath = "/";
    }

    while ((pos = path.find_first_of('/', pos)) != std::string::npos) {
        currentPath += path.substr(currentPath.length(), pos - currentPath.length() + 1);
        if (currentPath.length() > 1 && access(currentPath.c_str(), F_OK) != 0) { // Directory does not exist
            if (mkdir(currentPath.c_str(), 0777) != 0 && errno != EEXIST) { // Try creating directory
                std::cerr << "Failed to create directory '" << currentPath << "': " << strerror(errno) << std::endl;
                return false;
            }
        }
        pos++;
    }

    // Create last segment if it ends without a '/'
    if (pos != path.length() + 1) {
        if (mkdir(path.c_str(), 0777) != 0 && errno != EEXIST) {
            std::cerr << "Failed to create directory '" << path << "': " << strerror(errno) << std::endl;
            return false;
        }
    }

    return true;
}

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
    if (argc < 7) {
        cerr << "Usage: " << argv[0] << " <seed> <num> <gamma> <length> <t_max> <dest_path>" << endl;
        return 1;
    }

    int seed = stoi(argv[1]);
    int num = stoi(argv[2]);
    double gamma = stod(argv[3]);
    int length = stoi(argv[4]);
    int t_max = stoi(argv[5]);
    string dest_path = argv[6];

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

    // Remove trailing '/' if it exists
    if (dest_path[-1] == '/') dest_path = dest_path.substr(0, dest_path.size() - 1);
    dest_path = dest_path + "/" + sim_base;

    // Check if the folder already exists
    if (access(dest_path.c_str(), F_OK) != 0) {
        // Folder doesn't exist, create it
        if (!createDirectoryRecursively(dest_path)) {
            std::cerr << "Error creating directory structure for: " << dest_path << std::endl;
            return 1;
        }
        std::cout << "Directories created successfully at: " << dest_path << std::endl;

    } else {
        std::cout << "Folder already exists." << std::endl;
    }


    // Save correlator data
    ofstream corr_file(dest_path + "/" + to_string(seed) + "_" + sim_base + ".txt");
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
