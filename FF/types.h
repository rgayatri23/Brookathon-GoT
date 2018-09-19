#ifndef _TYPES_H
#define _TYPES_H
#include <thrust/complex.h>

typedef float REAL;
//typedef double REAL;
using GPUComplexD = thrust::complex<double>;
using GPUComplexF = thrust::complex<float>;
using GPUComplex = thrust::complex<REAL>;

#endif