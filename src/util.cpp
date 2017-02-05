/**
 * @file src/util.cpp
 * @date 2017/01/16 23:06:59
 * @brief 
 *  
 **/

#include "../include/util.h"
#include <chrono>
#include <random>

namespace dnn {

double sigmoid(double x) {
     return 1.0 / (1 + exp(-x)); 
}

double sigmoid_derivative(double x) {
     return sigmoid(x) * (1 - sigmoid(x)); 
}


// double tanh(double x) {
//      return 2 * sigmoid(2 * x) - 1; 
// }

double tanh_derivative(double x) {
     return 1 - pow(tanh(x), 2.0); 
}

void random_init(int length, double bound, std::vector<double>& values) {
    srand((unsigned)time(NULL));
    values.clear();
    for (int i = 0; i < length; i++) {
        // Init in [-bound, bound] range.
        double value = ((double) rand() / RAND_MAX) * (2 * bound) - bound;
        values.push_back(value);
    }
}

void normal_init(int length, double mean, double stddev, std::vector<double>& values) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);

    std::normal_distribution<double> distribution (mean, stddev);

    values.clear();
    for (int i = 0; i < length; i++) {
        values.push_back(distribution(generator));
    }
}

} // namespace dnn

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
