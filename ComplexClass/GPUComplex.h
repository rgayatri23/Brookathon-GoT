#ifndef __GPUComplex
#define __GPUComplex

#include <iostream>
#include <cstdlib>
#include <memory>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <ctime>
#include <stdio.h>
#include <sys/time.h>
#include <chrono>
#include <vector_types.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

using namespace std;

class GPUComplex : public double2{

    private : 
//    double x;
//    double y;

    public:
    explicit GPUComplex () {
        x = 0.00;
        y = 0.00;
    }


    __host__ __device__ explicit GPUComplex(const double& a, const double& b) {
        x = a;
        y = b;
    }

    __host__ __device__ GPUComplex(const GPUComplex& src) {
        x = src.x;
        y = src.y;
    }

    __host__ __device__ GPUComplex& operator =(const GPUComplex& src) {
        x = src.x;
        y = src.y;

        return *this;
    }

    __host__ __device__ GPUComplex& operator +=(const GPUComplex& src) {
        x = src.x + this->x;
        y = src.y + this->y;

        return *this;
    }

    __host__ __device__ GPUComplex& operator -=(const GPUComplex& src) {
        x = src.x - this->x;
        y = src.y - this->y;

        return *this;
    }

    __host__ __device__ GPUComplex& operator -() {
        x = -this->x;
        y = -this->y;

        return *this;
    }

    GPUComplex& operator ~() {
        return *this;
    }

    __host__ __device__ void print() const {
        printf("( %f, %f) ", this->x, this->y);
        printf("\n");
    }

    double get_real() const
    {
        return this->x;
    }

    double get_imag() const
    {
        return this->y;
    }

    __host__ __device__     void set_real(double val)
    {
        this->x = val;
    }

    __host__ __device__     void set_imag(double val) 
    {
        this->y = val;
    }

// 6 flops
//    template<class real, class imag>
    __host__ __device__ friend inline GPUComplex operator *(const GPUComplex &a, const GPUComplex &b) {
        double x_this = a.x * b.x - a.y*b.y ;
        double y_this = a.x * b.y + a.y*b.x ;
        GPUComplex result(x_this, y_this);
        return (result);
    }

//2 flops
//    template<class real, class imag>
    __host__ __device__ friend inline GPUComplex operator *(const GPUComplex &a, const double &b) {
       GPUComplex result(a.x*b, a.y*b);
       return result;
    }

//    template<class real, class imag>
    __host__ __device__ friend inline GPUComplex operator -(GPUComplex a, GPUComplex b) {
        GPUComplex result(a.x - b.x, a.y - b.y);
        return result;
    }

//2 flops
//    template<class real, class imag>
    __host__ __device__ friend inline GPUComplex operator -(const double &a, GPUComplex& src) {
        GPUComplex result(a - src.x, 0 - src.y);
        return result;
    }

    //template<class real, class imag>
    __host__ __device__ friend inline GPUComplex operator +(const double &a, GPUComplex& src) {
        GPUComplex result(a + src.x, src.y);
        return result;
    }

    //template<class real, class imag>
    __host__ __device__ friend inline GPUComplex operator +(GPUComplex a, GPUComplex b) {
        GPUComplex result(a.x + b.x, a.y+b.y);
        return result;
    }

    //template<class real, class imag>
    __host__ __device__ friend inline GPUComplex operator /(GPUComplex a, GPUComplex b) {

        GPUComplex b_conj = GPUComplex_conj(b);
        GPUComplex numerator = a * b_conj;
        GPUComplex denominator = b * b_conj;

        double re_this = numerator.x / denominator.x;
        double im_this = numerator.y / denominator.x;

        GPUComplex result(re_this, im_this);
        return result;
    }

    //template<class real, class imag>
    __host__ __device__ friend inline GPUComplex operator /(GPUComplex a, double b) {
       GPUComplex result(a.x/b, a.y/b);
       return result;
    }

    //template<class real, class imag>
    __host__ __device__ friend inline GPUComplex GPUComplex_conj(const GPUComplex& src) ;

    //template<class real, class imag>
    __host__ __device__ friend inline double GPUComplex_abs(const GPUComplex& src) ;

    //template<class real, class imag>
    __host__ __device__ friend inline double GPUComplex_real( const GPUComplex& src) ;

    //template<class real, class imag>
    __host__ __device__ friend inline double GPUComplex_imag( const GPUComplex& src) ;
};

/*
 * Return the conjugate of a complex number 
 1flop
 */
//template<class re, class im>
inline GPUComplex GPUComplex_conj(const GPUComplex& src) {

    double re_this = src.x;
    double im_this = -1 * src.y;

    GPUComplex result(re_this, im_this);
    return result;

}

/*
 * Return the absolute of a complex number 
 */
//template<class re, class im>
inline double GPUComplex_abs(const GPUComplex& src) {
    double re_this = src.x * src.x;
    double im_this = src.y * src.y;

    double result = sqrt(re_this+im_this);
    return result;
}

/*
 * Return the real part of a complex number 
 */
//template<class re, class im>
inline double GPUComplex_real( const GPUComplex& src) {
    return src.x;
}

/*
 * Return the imaginary part of a complex number 
 */
//template<class re, class im>
inline double GPUComplex_imag( const GPUComplex& src) {
    return src.y;
}

#endif


//GPP Function definition
void noflagOCC_cudaKernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, GPUComplex *wtilde_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im);

//FF Function definition
inline void schDttt_corKernel1(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex &schDttt, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2);

inline void schDttt_corKernel2(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2);

void d_achsDtemp_Kernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, double *vcoul, double *achsDtemp_re, double *achsDtemp_im);

void d_asxDtemp_Kernel(int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double occ, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, double *asxDtemp_re, double *asxDtemp_im);

void d_achDtemp_cor_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *ach2Dtemp, double *achDtemp_cor_re, double *achDtemp_cor_im, GPUComplex *achDtemp_corb);
