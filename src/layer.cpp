/**
 * @file src/layer.cpp
 * @date 2017/01/18 10:32:40
 * @brief 
 *  
 **/

#include "../include/layer.h"
#include <cassert>
#include <stddef.h>
#include <stdio.h> 
#include "../include/neuron.h"

namespace dnn {

Layer::Layer(const std::string& name, int neuron_count, Layer* prev, double (*activation)(double), 
        double (*activation_derivative)(double)) :
    _name(name), _prev(prev), _next(NULL) {
    printf("construct layer %s, neuron count is %d\n", name.c_str(), neuron_count);
    assert(_prev != NULL);
    _input_count = _prev->size();

    init_neurons(neuron_count, _input_count, activation, activation_derivative);

}


Layer::Layer(const std::string& name, int neuron_count, int input_count, double (*activation)(double), 
        double (*activation_derivative)(double)) : 
    _name(name), _prev(NULL), _next(NULL), _input_count(input_count) {
    assert(input_count > 0);
    printf("construct layer %s, input count is %d, neuron count is %d\n", 
            name.c_str(), input_count, neuron_count);
    init_neurons(neuron_count, _input_count, activation, activation_derivative);
}

Layer::~Layer() {
    printf("deconstruct layer %s\n", _name.c_str());
    for (unsigned int i = 0; i < _neurons.size(); i++) {
        delete _neurons[i];
    }
    _neurons.clear();
}


void Layer::init_neurons(int neuron_count, int input_count, double (*activation)(double), 
        double (*activation_derivative)(double)) {
    for (int i = 0; i < neuron_count; i++) {
        _neurons.push_back(new Neuron(input_count, activation, activation_derivative));
    }
}

void Layer::set_next_layer(Layer* next) {
    _next = next;
}


void Layer::forward(const std::vector<double>& inputs) {
    _outputs.clear();
    _s.clear();
    for (std::vector<Neuron*>::iterator it = _neurons.begin(); it != _neurons.end(); it++) {
        (*it)->forward(inputs);
        _outputs.push_back((*it)->output());
        _s.push_back((*it)->s());
    }

    if (_next != NULL) {
        _next->forward(_outputs);
    }
}

void Layer::backward(const std::vector<double>& delta_w_sums) {
    printf("layer %s backward\n", _name.c_str());
    assert(delta_w_sums.size() == _neurons.size());
    std::vector<double> current_delta_w_sums(_input_count, 0.0);
    for (unsigned int i = 0; i < _neurons.size(); i++) {
        _neurons[i]->backward(delta_w_sums[i]);
        for (int j = 0; j < _input_count; j++) {
            current_delta_w_sums[j] += _neurons[i]->weights()[j] * _neurons[i]->delta();
        }
    }

    if (_prev != NULL) {
        _prev->backward(current_delta_w_sums);
    }
}

void Layer::update_diff(const std::vector<double>& inputs) {
    for (unsigned int i = 0; i < _neurons.size(); i++) {
        _neurons[i]->update_diff(inputs);
    }

    if (_next != NULL) {
        _next->update_diff(_outputs);
    }
}

void Layer::gradient_descent(double learning_rate) {
    for (unsigned int i = 0; i < _neurons.size(); i++) {
        _neurons[i]->gradient_descent(learning_rate);
    }

    if (_next != NULL) {
        _next->gradient_descent(learning_rate);
    }
}




} // namespace dnn

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
