/**
 * @file neuron.h
 * @date 2017/01/15 18:55:17
 * @brief 
 *  
 **/

#ifndef  __NEURON_H_
#define  __NEURON_H_

#include <vector>

namespace dnn {

class Neuron {
public:
    Neuron(int input_size, double (*activation)(double), 
            double (*activation_derivative)(double));

    double output() const {
        return _output;
    }

    double s() const {
        return _s;
    }

    double delta() const {
        return _delta;
    }

    void set_delta(double delta) {
        _delta = delta;
    }

    const std::vector<double>& weights() const {
        return _weights;
    }

    void forward(const std::vector<double>& inputs);

    void backward(double delta_w_sum);

    void update_diff(const std::vector<double>& inputs);

    void gradient_descent(double learning_rate);
    

private:
    double (*_activation)(double);
    double (*_activation_derivative)(double);
    std::vector<double> _weights;
    std::vector<std::vector<double>> _diffs;
    double _s;
    double _output;
    double _delta;

};

} // namespace dnn

#endif  //__NEURON_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
