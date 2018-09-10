#include "../ComplexClass/GPUComplex.h"
#define nstart 0
#define nend 3

/*
 * Return the square of a complex number 
 */
__device__ inline GPUComplex d_GPUComplex_square(GPUComplex& src) {
    return GPUComplex(src.x*src.x - src.y*src.y, 2*src.x*src.y);
}

/*
 * Return the conjugate of a complex number 
 */
__device__ inline GPUComplex d_GPUComplex_conj(const GPUComplex& src) {
return GPUComplex(src.x, -src.y);
}


/*
 * Return the product of 2 complex numbers 
 */
__device__ inline GPUComplex d_GPUComplex_product(const GPUComplex& a, const GPUComplex& b) {
    return GPUComplex(a.x * b.x - a.y*b.y, a.x * b.y + a.y*b.x);
}


/*
 * Return the absolute of a complex number 
 */
__device__ inline double d_GPUComplex_abs(const GPUComplex& src) {
    return sqrt(src.x * src.x + src.y * src.y);
}

/*
 *  result = a * b * c (a = complex ; b,c = double) 
 */
__device__ inline GPUComplex d_GPUComplex_mult(GPUComplex& a, double b, double c) {
    return GPUComplex(a.x * b * c, a.y * b * c);
}

/*
 * Return the complex number c = a * b (a is complex, b is double) 
 */
__device__ inline GPUComplex d_GPUComplex_mult(const GPUComplex& a, double b) {
   return GPUComplex(a.x*b, a.y*b);

}

/*
 * Return the complex number a += b * c  
 */
__device__ inline void d_GPUComplex_fma(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    a.x += b.x * c.x - b.y*c.y ;
    a.y += b.x * c.y + b.y*c.x ;
}

/*
 * Return the complex number a -= b * c  
 */
__device__ inline void d_GPUComplex_fms(GPUComplex& a, const GPUComplex& b, const GPUComplex& c) {
    a.x -= b.x * c.x - b.y*c.y ;
    a.y -= b.x * c.y + b.y*c.x ;
}


__device__ inline GPUComplex d_doubleMinusGPUComplex(const double &a, GPUComplex& src) {
    return GPUComplex(a-src.x, -src.y);
}

__device__ inline GPUComplex d_doublePlusGPUComplex(double a, GPUComplex& src) {
    return GPUComplex(a+src.x, src.y);
}

__device__ inline double d_GPUComplex_real( const GPUComplex& src) {
    return src.x;
}

__device__ inline double d_GPUComplex_imag( const GPUComplex& src) {
    return src.y;
}

__device__ inline void d_GPUComplex_plusEquals( GPUComplex& a, const GPUComplex & b) {
    a.x += b.x;
    a.y += b.y;
}

__device__ void inline d_GPUComplex_Equals( GPUComplex& a, const GPUComplex & b) {
    a.x = b.x;
    a.y = b.y;
}

__device__ void d_print( const GPUComplex& a) {
    printf("( %f, %f) ", a.x, a.y);
    printf("\n");
}

__device__ inline void ncouls_Kernel(double wx_array_index, GPUComplex wtilde_array_index, GPUComplex I_eps_array_index, GPUComplex aqsmtemp_index, GPUComplex aqsntemp_index, double vcoul_index, double &achtemp_re_iw, double &achtemp_im_iw)
{
    GPUComplex wdiff = wx_array_index - wtilde_array_index; //2 flops
    GPUComplex delw = wtilde_array_index * GPUComplex_conj(wdiff) * (1/GPUComplex_real((wdiff * GPUComplex_conj(wdiff)))); 
    GPUComplex sch_array = GPUComplex_conj(aqsmtemp_index) * aqsntemp_index * delw * I_eps_array_index * 0.5*vcoul_index;

    //2 flops
    achtemp_re_iw += GPUComplex_real(sch_array);
    achtemp_im_iw += GPUComplex_imag(sch_array);
}

__global__ void gpp_2D_CUDAKernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, GPUComplex *wtilde_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im, int numThreadsPerBlock)
{
    int n1 = blockIdx.x;
    int my_igp = blockIdx.y;

    if((n1 < number_bands ) && (my_igp < ngpown) )
    {
        int loopOverncouls = 1, leftOverncouls = 0;
        if(ncouls > numThreadsPerBlock)
        {
            loopOverncouls = ncouls / numThreadsPerBlock;
            leftOverncouls = ncouls % numThreadsPerBlock;
        }

        double achtemp_re_loc[nend-nstart], achtemp_im_loc[nend-nstart];
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];
        for(int iw = nstart; iw < nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}

        for( int x = 0; x < loopOverncouls && threadIdx.x < numThreadsPerBlock ; ++x)
        { 
            int ig = x*numThreadsPerBlock + threadIdx.x;
            if(ig < ncouls)
            {
                for(int iw = nstart; iw < nend; ++iw)
                    ncouls_Kernel(wx_array[iw], wtilde_array[my_igp*ncouls +ig], I_eps_array[my_igp*ncouls +ig], aqsmtemp[n1*ncouls +igp], aqsntemp[n1*ncouls +ig], vcoul[igp], achtemp_re_loc[iw], achtemp_im_loc[iw]);
            }
        }
        if(leftOverncouls)
        {
            int ig = loopOverncouls*numThreadsPerBlock + threadIdx.x;
            if(ig < ncouls)
            {
                for(int iw = nstart; iw < nend; ++iw)
                    ncouls_Kernel(wx_array[iw], wtilde_array[my_igp*ncouls +ig], I_eps_array[my_igp*ncouls +ig], aqsmtemp[n1*ncouls +igp], aqsntemp[n1*ncouls +ig], vcoul[igp], achtemp_re_loc[iw], achtemp_im_loc[iw]);
            }
        }

        for(int iw = nstart; iw < nend; ++iw)
        {
            atomicAdd(&achtemp_re[iw] , achtemp_re_loc[iw] );
            atomicAdd(&achtemp_im[iw] , achtemp_im_loc[iw] );
        }
    }
}

void noflagOCC_cudaKernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, GPUComplex *wtilde_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im)
{
    dim3 numBlocks(number_bands, ngpown);
    int numThreadsPerBlock = 32;
    printf("Launching a double dimension grid with numBlocks = (%d, %d) and %d threadsPerBlock \n", number_bands, ngpown, numThreadsPerBlock);

    gpp_2D_CUDAKernel<<<numBlocks, numThreadsPerBlock>>> (number_bands, ngpown, ncouls, inv_igp_index, indinv, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, vcoul, achtemp_re, achtemp_im, numThreadsPerBlock);
}


