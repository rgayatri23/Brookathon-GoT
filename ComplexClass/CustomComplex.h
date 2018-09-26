#ifndef __CustomComplex
#define __CustomComplex

#include <iostream>
#include <cstdlib>
#include <memory>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <ctime>
#include <stdio.h>
#include <sys/time.h>
#include <vector_types.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
using namespace std;

template<class type>

class CustomComplex {

    private : 
//    type x;
//    type y;

    public:
    type x;
    type y;
    explicit CustomComplex () {
        x = 0.00;
        y = 0.00;
    }


    __host__ __device__ explicit CustomComplex(const double& a, const double& b) {
        x = a;
        y = b;
    }

    __host__ __device__ CustomComplex(const CustomComplex& src) {
        x = src.x;
        y = src.y;
    }

    __host__ __device__ CustomComplex& operator =(const CustomComplex& src) {
        x = src.x;
        y = src.y;

        return *this;
    }

    __host__ __device__ CustomComplex& operator +=(const CustomComplex& src) {
        x = src.x + this->x;
        y = src.y + this->y;

        return *this;
    }

    __host__ __device__ CustomComplex& operator -=(const CustomComplex& src) {
        x = src.x - this->x;
        y = src.y - this->y;

        return *this;
    }

    __host__ __device__ CustomComplex& operator -() {
        x = -this->x;
        y = -this->y;

        return *this;
    }

    CustomComplex& operator ~() {
        return *this;
    }

    void print() const {
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

    void set_real(double val)
    {
        this->x = val;
    }

    void set_imag(double val) 
    {
        this->y = val;
    }

// 6 flops
    template<class T>
    __host__ __device__ friend inline CustomComplex<T> operator *(const CustomComplex<T> &a, const CustomComplex<T> &b) {
        T x_this = a.x * b.x - a.y*b.y ;
        T y_this = a.x * b.y + a.y*b.x ;
        CustomComplex<T> result(x_this, y_this);
        return (result);
    }

//2 flops
    template<class T>
    __host__ __device__ friend inline CustomComplex<T> operator *(const CustomComplex<T> &a, const double &b) {
       CustomComplex<T> result(a.x*b, a.y*b);
       return result;
    }

    template<class T>
    __host__ __device__ friend inline CustomComplex<T> operator -(CustomComplex<T> a, CustomComplex<T> b) {
        CustomComplex<T> result(a.x - b.x, a.y - b.y);
        return result;
    }

//2 flops
    template<class T>
    __host__ __device__ friend inline CustomComplex<T> operator -(const double &a, CustomComplex<T>& src) {
        CustomComplex<T> result(a - src.x, 0 - src.y);
        return result;
    }

    template<class T>
    __host__ __device__ friend inline CustomComplex<T> operator +(const double &a, CustomComplex<T>& src) {
        CustomComplex<T> result(a + src.x, src.y);
        return result;
    }

    template<class T>
    __host__ __device__ friend inline CustomComplex<T> operator +(CustomComplex<T> a, CustomComplex<T> b) {
        CustomComplex<T> result(a.x + b.x, a.y+b.y);
        return result;
    }

    template<class T>
    __host__ __device__ friend inline CustomComplex<T> operator /(CustomComplex<T> a, CustomComplex<T> b) {

        CustomComplex<T> b_conj = CustomComplex_conj(b);
        CustomComplex<T> numerator = a * b_conj;
        CustomComplex<T> denominator = b * b_conj;

        double re_this = numerator.x / denominator.x;
        double im_this = numerator.y / denominator.x;

        CustomComplex<T> result(re_this, im_this);
        return result;
    }

    template<class T>
    __host__ __device__ friend inline CustomComplex<T> operator /(CustomComplex<T> a, T b) {
       CustomComplex<T> result(a.x/b, a.y/b);
       return result;
    }

    template<class T>
    __host__ __device__ friend inline CustomComplex<T> CustomComplex_conj(const CustomComplex<T>& src) ;

    template<class T>
    __host__ __device__ friend inline double CustomComplex_abs(const CustomComplex<T>& src) ;

    template<class T>
    __host__ __device__ friend inline double CustomComplex_real( const CustomComplex<T>& src) ;

    template<class T>
    __host__ __device__ friend inline double CustomComplex_imag( const CustomComplex<T>& src) ;

};


/* Return the conjugate of a complex number 
flop
*/
template<class T>
inline CustomComplex<T> CustomComplex_conj(const CustomComplex<T>& src) {

    T re_this = src.x;
    T im_this = -1 * src.y;

    CustomComplex<T> result(re_this, im_this);
    return result;

}

/*
 * Return the absolute of a complex number 
 */
template<class T>
inline double CustomComplex_abs(const CustomComplex<T>& src) {
    T re_this = src.x * src.x;
    T im_this = src.y * src.y;

    T result = sqrt(re_this+im_this);
    return result;
}

/*
 * Return the real part of a complex number 
 */
template<class T>
inline double CustomComplex_real( const CustomComplex<T>& src) {
    return src.x;
}

/*
 * Return the imaginary part of a complex number 
 */
template<class T>
inline double CustomComplex_imag( const CustomComplex<T>& src) {
    return src.y;
}

//GPP Function definition
template<class T>
void noflagOCC_cudaKernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, CustomComplex<double> *wtilde_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, CustomComplex<double> *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im);

//FF Function definition
template<class T>
inline void schDttt_corKernel1(CustomComplex<double> &schDttt_cor, int *inv_igp_index, int *indinv, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, CustomComplex<double> &schDttt, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2);

template<class T>
inline void schDttt_corKernel2(CustomComplex<double> &schDttt_cor, int *inv_igp_index, int *indinv, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2);

//template<class T>
void d_achsDtemp_Kernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, CustomComplex<double> *aqsntemp, CustomComplex<double> *aqsmtemp, CustomComplex<double> *I_epsR_array, double *vcoul, double *achsDtemp_re, double *achsDtemp_im);

template<class T>
void d_asxDtemp_Kernel(int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double occ, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, double *asxDtemp_re, double *asxDtemp_im);

void d_achDtemp_cor_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *ach2Dtemp, double *achDtemp_cor_re, double *achDtemp_cor_im, CustomComplex<double> *achDtemp_corb);
#endif
