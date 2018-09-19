#ifndef _TYPES_H
#define _TYPES_H
#include <thrust/complex.h>

typedef float REAL;
//typedef double REAL;
using GPUComplex = thrust::complex<REAL>;
#endif