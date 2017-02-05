/***************************************************************************
 * 
 * Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file src/loss_layer.cpp
 * @date 2017/02/04 11:04:00
 * @brief 
 *  
 **/

#include "../include/loss_layer.h"
#include <cassert>
#include <stddef.h>
#include <stdio.h> 

namespace dnn {

LossLayer::LossLayer(const std::string& name, int neuron_count, Layer* prev, double (*activation)(double), 
        double (*activation_derivative)(double)) :
    Layer(name, neuron_count, prev, activation, activation_derivative) {
        printf("above is the loss(output) layer\n");
}


void LossLayer::backward(const std::vector<double>& y) {
    assert(y.size() == _neurons.size());
    std::vector<double> current_delta_w_sums(_input_count);
    printf("loss is: ");
    // use square loss (y-f(x))^2/2, whose gradient is f(x) - y
    for (unsigned int i = 0; i < _neurons.size(); i++) {
        _neurons[i]->set_delta(_s[i] - y[i]);
        printf("%f\t", _s[i] - y[i]);
        for (int j = 0; j < _input_count; j++) {
            current_delta_w_sums[j] += _neurons[i]->weights()[j] * _neurons[i]->delta();
        }
    }
    printf("\n");

    if (_prev != NULL) {
        _prev->backward(current_delta_w_sums);
    }
}


} // namespace dnn

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
