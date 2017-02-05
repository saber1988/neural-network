/**
 * @file src/net.cpp
 * @date 2017/01/18 19:25:49
 * @brief 
 *  
 **/

#include "../include/net.h"
#include <cassert>
#include <stdio.h> 
#include "../include/loss_layer.h"
#include "../include/util.h"

namespace dnn {

Net::Net(const std::vector<int>& layers, const std::vector<std::string>& names) {
    printf("construct net\n");
    assert(layers.size() > 2);
    assert(layers.size() == (names.size() + 1));
    // first layer, including input
    Layer* prev = new Layer(names[0], layers[1], layers[0], &sigmoid, &sigmoid_derivative);

    for (unsigned int i = 2; i < layers.size() - 1; i++) {
        Layer* current = new Layer(names[i - 1], layers[i], prev, &sigmoid, &sigmoid_derivative);
        prev->set_next_layer(current);

        _layers.push_back(prev);

        prev = current;
    }

    // output(loss) layer
    Layer* current = new LossLayer(names[names.size() - 1], layers[layers.size() - 1], prev, &sigmoid, &sigmoid_derivative);
    prev->set_next_layer(current);

    _layers.push_back(prev);
    _layers.push_back(current);
}

Net::~Net() {
    printf("deconstruct net\n");
    for (unsigned int i = 0; i < _layers.size(); i++) {
         delete _layers[i];
    }

}

void Net::train_one_batch(const std::vector<std::vector<double> >& x, const std::vector<std::vector<double> >& y, double learning_rate) {
    assert(x.size() == y.size());
    for (unsigned int i = 0; i < x.size(); i++) {
        forward_backward(x[i], y[i]);
    }
    update(learning_rate);

}


void Net::predict(const std::vector<double>& x, std::vector<double>& y) const {
    y.clear();
    _layers[0]->forward(x);
    Layer* output_layer = _layers[_layers.size() -1];
    for (unsigned int i = 0; i < output_layer->s().size() ; i++) {
        y.push_back(output_layer->s()[i]);
    }

}


void Net::forward_backward(const std::vector<double>& x, const std::vector<double>& y) {
    _layers[0]->forward(x);
    _layers[_layers.size() -1]->backward(y);
    _layers[0]->update_diff(x);
}


void Net::update(double learning_rate) {
    _layers[0]->gradient_descent(learning_rate);

}









} // namespace dnn

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
