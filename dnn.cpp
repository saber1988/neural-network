#include <iostream>
#include <vector>
#include <stdio.h> 

#include "include/util.h"
#include "include/neuron.h"
#include "include/net.h"

using namespace dnn;

int main() {

    printf("begin to build net.\n");
    std::vector<int> layers = {3, 5, 2};
    std::vector<std::string> names = {"FullConnectedLayer1", "SquareLossLayer"};
    std::vector<std::vector<double>> x = {{1.0, 2.0, 3.0}, {1.5, 0.5, 4.5}};
    std::vector<std::vector<double>> y = {{6.0, -4.0}, {6.5, -3.5}};
    Net net(layers, names);
    net.train_one_batch(x, y, 0.1);

    std::vector<double> input = {2.3, 3.1, 0.2};
    std::vector<double> output;
    net.predict(input, output);
    printf("predict result is: ");
    for (unsigned int i = 0; i < output.size(); i ++) {
        printf("%f\t", output[i]);
    }
    printf("\n");


    return 0;
}




/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
