/**
 * @file src/neuron.cpp
 * @date 2017/01/16 20:46:36
 * @brief 
 *  
 **/

#include "../include/neuron.h"
#include <cassert>
#include "../include/util.h"

namespace dnn {

Neuron::Neuron(int input_size, double (*activation)(double), double (*activation_derivative)(double)) : 
    _activation(activation), _activation_derivative(activation_derivative) {
        // Normalize the variance of each neuron's output to 1 by scaling its weight vector by the
        // square root of its number of inputs.
        // + 1 is for bias
        normal_init(input_size + 1, 0.0, 1.0 / sqrt(input_size), _weights);

}


void Neuron::forward(const std::vector<double>& inputs) {
    assert(inputs.size() + 1 == _weights.size());
    _s = _weights.back();
    for (unsigned int i = 0; i < inputs.size(); i++) {
        _s += inputs[i] * _weights[i];
    }
    _output = _activation(_s);
    

}

void Neuron::backward(double delta_w_sum) {
    _delta = _activation_derivative(_s) * delta_w_sum;
}

void Neuron::update_diff(const std::vector<double>& inputs) {
    assert(inputs.size() + 1 == _weights.size());
    std::vector<double> diff;
    for(unsigned int i = 0; i < inputs.size(); i++) {
        diff.push_back(inputs[i] * _delta);
    }
    diff.push_back(_delta);
    _diffs.push_back(diff);

}

void Neuron::gradient_descent(double learning_rate) {
    unsigned int batch_size = _diffs.size();
    assert(batch_size > 0);

    for(unsigned int i = 0; i < _weights.size(); i++) {
        double diff = 0.0;
        for (unsigned int j = 0; j < batch_size; j++) {
            diff += _diffs[j][i];
        }
        diff /= batch_size;
        _weights[i] -= learning_rate * diff;
    }
    _diffs.clear();
}


} // namespace dnn


/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
