/**
 * @file net.h
 * @date 2017/01/15 18:56:59
 * @brief 
 *  
 **/

#ifndef  __NET_H_
#define  __NET_H_

#include "../include/layer.h"

namespace dnn {

class Net {
public:
    Net(const std::vector<int>& layers, const std::vector<std::string>& names);
    ~Net();

    void train_one_batch(const std::vector<std::vector<double> >& x, const std::vector<std::vector<double> >& y, double learning_rate);

    void predict(const std::vector<double>& x, std::vector<double>& y) const;

private:
    void forward_backward(const std::vector<double>& x, const std::vector<double>& y);
    void update(double learning_rate);
    std::vector<Layer*> _layers;

};

} // namespace dnn

#endif  //__NET_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
