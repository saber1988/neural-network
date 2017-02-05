/**
 * @file include/loss_layer.h
 * @date 2017/02/04 10:43:32
 * @brief 
 *  
 **/

#ifndef  __LOSS_LAYER_H_
#define  __LOSS_LAYER_H_

#include "../include/layer.h"

namespace dnn {

class LossLayer : public Layer {

public:
    LossLayer(const std::string& name, int neuron_count, Layer* prev, double (*activation)(double), 
            double (*activation_derivative)(double));
    virtual void backward(const std::vector<double>& y);

};

} // namespace dnn

#endif  //__LOSS_LAYER_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
