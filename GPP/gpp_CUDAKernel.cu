#include "../ComplexClass/CustomComplex.h"
#define nstart 0
#define nend 3

template void noflagOCC_cudaKernel<double>(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, CustomComplex<double> *wtilde_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, CustomComplex<double> *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im);
/*
 * Return the square of a complex number
 */
__device__ inline CustomComplex<double> d_CustomComplex_square(CustomComplex<double>& src) {
    return CustomComplex<double>(src.x*src.x - src.y*src.y, 2*src.x*src.y);
}

/*
 * Return the conjugate of a complex number
 */
__device__ inline CustomComplex<double> d_CustomComplex_conj(const CustomComplex<double>& src) {
return CustomComplex<double>(src.x, -src.y);
}


/*
 * Return the product of 2 complex numbers
 */
__device__ inline CustomComplex<double> d_CustomComplex_product(const CustomComplex<double>& a, const CustomComplex<double>& b) {
    return CustomComplex<double>(a.x * b.x - a.y*b.y, a.x * b.y + a.y*b.x);
}


/*
 * Return the absolute of a complex number
 */
__device__ inline double d_CustomComplex_abs(const CustomComplex<double>& src) {
    return sqrt(src.x * src.x + src.y * src.y);
}

/*
 *  result = a * b * c (a = complex ; b,c = double)
 */
__device__ inline CustomComplex<double> d_CustomComplex_mult(CustomComplex<double>& a, double b, double c) {
    return CustomComplex<double>(a.x * b * c, a.y * b * c);
}

/*
 * Return the complex number c = a * b (a is complex, b is double)
 */
__device__ inline CustomComplex<double> d_CustomComplex_mult(const CustomComplex<double>& a, double b) {
   return CustomComplex<double>(a.x*b, a.y*b);

}

/*
 * Return the complex number a += b * c
 */
__device__ inline void d_CustomComplex_fma(CustomComplex<double>& a, const CustomComplex<double>& b, const CustomComplex<double>& c) {
    a.x += b.x * c.x - b.y*c.y ;
    a.y += b.x * c.y + b.y*c.x ;
}

/*
 * Return the complex number a -= b * c
 */
__device__ inline void d_CustomComplex_fms(CustomComplex<double>& a, const CustomComplex<double>& b, const CustomComplex<double>& c) {
    a.x -= b.x * c.x - b.y*c.y ;
    a.y -= b.x * c.y + b.y*c.x ;
}

template<class T>
__device__ inline CustomComplex<T> d_doubleMinusCustomComplex(const double &a, CustomComplex<double>& src) {
    return CustomComplex<T>(a-src.x, -src.y);
}

template<class T>
__device__ inline CustomComplex<T> d_doublePlusCustomComplex(double a, CustomComplex<double>& src) {
    return CustomComplex<T>(a+src.x, src.y);
}

__device__ inline double d_CustomComplex_real( const CustomComplex<double>& src) {
    return src.x;
}

__device__ inline double d_CustomComplex_imag( const CustomComplex<double>& src) {
    return src.y;
}

__device__ inline void d_CustomComplex_plusEquals( CustomComplex<double>& a, const CustomComplex<double> & b) {
    a.x += b.x;
    a.y += b.y;
}

__device__ void inline d_CustomComplex_Equals( CustomComplex<double>& a, const CustomComplex<double> & b) {
    a.x = b.x;
    a.y = b.y;
}

__device__ void d_print( const CustomComplex<double>& a) {
    printf("( %f, %f) ", a.x, a.y);
    printf("\n");
}

__device__ inline void ncouls_Kernel(double wx_array_index, CustomComplex<double> wtilde_array_index, CustomComplex<double> I_eps_array_index, CustomComplex<double> aqsmtemp_index, CustomComplex<double> aqsntemp_index, double vcoul_index, double &achtemp_re_iw, double &achtemp_im_iw)
{
    CustomComplex<double> wdiff = wx_array_index - wtilde_array_index; //2 flops
    CustomComplex<double> delw = wtilde_array_index * CustomComplex_conj(wdiff) * (1/CustomComplex_real((wdiff * CustomComplex_conj(wdiff))));
    CustomComplex<double> sch_array = CustomComplex_conj(aqsmtemp_index) * aqsntemp_index * delw * I_eps_array_index * 0.5*vcoul_index;

    //2 flops
    achtemp_re_iw += CustomComplex_real(sch_array);
    achtemp_im_iw += CustomComplex_imag(sch_array);
}

__global__ void gpp_2D_CUDAKernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, CustomComplex<double> *wtilde_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, CustomComplex<double> *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im, int numThreadsPerBlock)
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

__global__ void gpp_2D_CUDAKernel_V2(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, CustomComplex<double> *wtilde_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, CustomComplex<double> *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im, int numThreadsPerBlock)
{
    CustomComplex<double> wdiff(0.00, 0.00), delw(0.00, 0.00), sch_array(0.00, 0.00);

    for(int n1 = blockIdx.x; n1 < number_bands; n1 += gridDim.x)
   {
        for(int my_igp = blockIdx.y; my_igp < ngpown; my_igp += gridDim.y)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
	    CustomComplex<double> aqsmtemp_vcoul = CustomComplex_conj(aqsmtemp[n1*ncouls+igp]) * 0.5*vcoul[igp];

	    double achtemp_re_loc[nend-nstart], achtemp_im_loc[nend-nstart];
            for(int iw = nstart; iw < nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}

            for(int ig = threadIdx.x; ig < ncouls; ig += blockDim.x)
            {
		CustomComplex<double> wtilde_ig = wtilde_array[my_igp*ncouls+ig];
		CustomComplex<double> sch_store1 = aqsntemp[n1*ncouls + ig] * I_eps_array[my_igp*ncouls+ig] * aqsmtemp_vcoul;
                for(int iw = nstart; iw < nend; ++iw)
                {
                    wdiff = wx_array[iw] - wtilde_array[my_igp*ncouls+ig];
                    delw = wtilde_ig * CustomComplex_conj(wdiff) * (1/CustomComplex_real((wdiff * CustomComplex_conj(wdiff))));
                    sch_array = sch_store1 * delw;

                    achtemp_re_loc[iw] += CustomComplex_real(sch_array);
                    achtemp_im_loc[iw] += CustomComplex_imag(sch_array);
                }
            }
	    //Add up the locally stored values
	    for(int iw = nstart; iw < nend; ++iw)
	    {
		atomicAdd(&achtemp_re[iw] , achtemp_re_loc[iw] );
		atomicAdd(&achtemp_im[iw] , achtemp_im_loc[iw] );
	    }
        } //ngpown
    } //number_bands

}

template<class T>
void noflagOCC_cudaKernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, CustomComplex<T> *wtilde_array, CustomComplex<T> *aqsmtemp, CustomComplex<T> *aqsntemp, CustomComplex<T> *I_eps_array, double *vcoul, T *achtemp_re, T *achtemp_im)
{
    dim3 numBlocks(number_bands, ngpown);
    int numThreadsPerBlock = 64;
    dim3 numThreads(32, 1, 1);

    printf("Launching a double dimension grid with numBlocks = (%d, %d) and %d threadsPerBlock \n", number_bands, ngpown, numThreadsPerBlock);

//    gpp_2D_CUDAKernel<<<numBlocks, numThreadsPerBlock>>> (number_bands, ngpown, ncouls, inv_igp_index, indinv, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, vcoul, achtemp_re, achtemp_im, numThreadsPerBlock);
    gpp_2D_CUDAKernel_V2<<<numBlocks,numThreads>>>(number_bands, ngpown, ncouls, inv_igp_index, indinv, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, vcoul, achtemp_re, achtemp_im, numThreadsPerBlock);
}


