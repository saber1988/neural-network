/**
 * @file layer.h
 * @date 2017/01/15 18:56:15
 * @brief 
 *  
 **/

#ifndef  __LAYER_H_
#define  __LAYER_H_

#include <vector>
#include <string>
#include "../include/neuron.h"

namespace dnn {

class Layer {
public:
    Layer(const std::string& name, int neuron_count, Layer* prev, double (*activation)(double), 
            double (*activation_derivative)(double));
    Layer(const std::string& name, int neuron_count, int input_count, double (*activation)(double), 
            double (*activation_derivative)(double));
    virtual ~Layer();

    void set_next_layer(Layer* next);

    int size() const { 
        return _neurons.size(); 
    }

    const std::vector<double>& s() const {
        return _s;
    }

    const int input_count() const {
        return _input_count;
    }


    void forward(const std::vector<double>& inputs);

    virtual void backward(const std::vector<double>& delta_w_sums);

    void update_diff(const std::vector<double>& inputs);

    void gradient_descent(double learning_rate);
    
protected:
    void init_neurons(int neuron_count, int input_count, double (*activation)(double), 
            double (*activation_derivative)(double));

    std::vector<double> _outputs;
    std::vector<double> _s;
    std::string _name;
    Layer* _prev;
    Layer* _next;
    int _input_count;
    std::vector<Neuron*> _neurons;

};

} // namespace dnn

#endif  //__LAYER_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
