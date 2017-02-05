#include <iostream>
#include <vector>
#include <stdio.h> 

#include "include/util.h"
#include "include/neuron.h"
#include "include/net.h"

using namespace dnn;

int main() {
    printf("sigmoid of 5.0 is %f.\n", sigmoid(5.0));
    printf("tanh of 5.0 is %f.\n", tanh(5.0));
    printf("sigmoid_derivative of 5.0 is %f.\n", sigmoid_derivative(5.0));
    printf("tanh_derivative of 5.0 is %f.\n", tanh_derivative(5.0));

    std::vector<double> values;
    random_init(10, 2.0, values);

    for (std::vector<double>::const_iterator it = values.begin(); it != values.end(); it++) {
        printf("random init, value is %f.\n", *it);
    }

    normal_init(10, 0.0, 1.0, values);

    for (std::vector<double>::const_iterator it = values.begin(); it != values.end(); it++) {
        printf("normal init, value is %f.\n", *it);
    }

    Neuron neuron(10, &sigmoid, &sigmoid_derivative);

    for (std::vector<double>::const_iterator it = neuron.weights().begin(); it != neuron.weights().end(); it++) {
        printf("weight is %f.\n", *it);
    }

    return 0;
}




/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
