/**
 * @file util.h
 * @date 2017/01/16 19:53:36
 * @brief 
 *  
 **/

#ifndef  __UTIL_H_
#define  __UTIL_H_

#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>

namespace dnn {

double sigmoid(double x);

double sigmoid_derivative(double x);


// double tanh(double x);

double tanh_derivative(double x);

void random_init(int length, double bound, std::vector<double>& values);

void normal_init(int length, double mean, double stddev, std::vector<double>& values);

} // namespace dnn

#endif  //__UTIL_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
