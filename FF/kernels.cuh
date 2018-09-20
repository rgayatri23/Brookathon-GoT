#ifndef _KERNELS_CUH
#define _KERNELS_CUH
#include "types.h"

// Atomic add operation for double
//#if defined( __CUDA_ARCH__ ) && __CUDA_ARCH__ >= 600
#define atomicAdd2 atomicAdd
//#else
//__device__ double atomicAdd2( double *address, double val )
//{
//    unsigned long long int *address_as_ull = (unsigned long long int *) address;
//    unsigned long long int old             = *address_as_ull, assumed;
//    do {
//        assumed = old;
//        old     = atomicCAS( address_as_ull, assumed,
//            __double_as_longlong( val + __longlong_as_double( assumed ) ) );
//    } while ( assumed != old );
//    return __longlong_as_double( old );
//}
//#endif

    template<int T>
__global__ void achsDtemp_solver_1D(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, REAL *vcoul, REAL *achsDtemp_re, REAL *achsDtemp_im)
{
    const int n1 = blockIdx.x;
    int numBlk = (ncouls+blockDim.x-1)/blockDim.x;
    typedef cub::BlockReduce<GPUComplex, T> BlockReduce;
    GPUComplex schsDtemp_red(0., 0.);
    __shared__ typename BlockReduce::TempStorage temp_storage;

    GPUComplex schsDtemp(0.00, 0.00);

    for (int my_igp = 0; my_igp < ngpown; ++my_igp)
    {
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];

        for (int blk = 0; blk < numBlk; blk++)
        {
            int ig = blk * blockDim.x + threadIdx.x;
            if (ig< ncouls)
                schsDtemp = schsDtemp - aqsntemp[n1*ncouls + ig] * thrust::conj(aqsmtemp[n1*ncouls + igp]) * I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig]* vcoul[ig] * 0.5;
        }
    }
    schsDtemp_red = BlockReduce(temp_storage).Sum(schsDtemp);

    if(threadIdx.x==0){
        atomicAdd2(achsDtemp_re, schsDtemp_red.real());
        atomicAdd2(achsDtemp_im, schsDtemp_red.imag());
    }
}

    template<int T>
__global__ void achsDtemp_solver_1D_mixed(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, REAL *vcoul, double *achsDtemp_re, double *achsDtemp_im)
{
    const int n1 = blockIdx.x;
    int numBlk = (ncouls+blockDim.x-1)/blockDim.x;
    typedef cub::BlockReduce<GPUComplexD, T> BlockReduce;
    GPUComplexD schsDtempD_red(0., 0.);
    __shared__ typename BlockReduce::TempStorage temp_storage;
    GPUComplex schsDtemp(0.00, 0.00);

    for(int my_igp = 0; my_igp < ngpown; ++my_igp){
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];

        for(int blk = 0; blk < numBlk; blk++)
        {
            int ig = blk * blockDim.x + threadIdx.x;
            if (ig< ncouls)
                schsDtemp = schsDtemp - aqsntemp[n1*ncouls + ig] * thrust::conj(aqsmtemp[n1*ncouls + igp]) * I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig]* vcoul[ig] * 0.5;
        }
    }
    GPUComplexD schsDtempD = GPUComplexD(static_cast<double>(schsDtemp.real()), static_cast<double>(schsDtemp.imag()));
    schsDtempD_red = BlockReduce(temp_storage).Sum(schsDtempD);

    if(threadIdx.x==0){
        atomicAdd2(achsDtemp_re, schsDtempD_red.real());
        atomicAdd2(achsDtemp_im, schsDtempD_red.imag());
    }
}


    template<int T, int S>
__global__ void achsDtemp_solver_2D(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, REAL *vcoul, REAL *achsDtemp_re, REAL *achsDtemp_im)
{
    //prepare redcution
    typedef cub::BlockReduce<GPUComplex, T, cub::BLOCK_REDUCE_WARP_REDUCTIONS, S> BlockReduce;

    GPUComplex schsDtemp(0.00, 0.00);

    const int n1 = blockIdx.y * blockDim.y + threadIdx.y;
    const int ig = blockIdx.x * blockDim.x + threadIdx.x;
    GPUComplex schsDtemp_red(0., 0.);
    __shared__ typename BlockReduce::TempStorage temp_storage;

    if( n1 < number_bands && ig < ncouls ){

        for(int my_igp = 0; my_igp < ngpown; ++my_igp){
            //do indirect access
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];

            schsDtemp = schsDtemp - aqsntemp[n1*ncouls + ig] * thrust::conj(aqsmtemp[n1*ncouls + igp]) * I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig]* vcoul[ig] * 0.5;
        }
    }

    schsDtemp_red = BlockReduce(temp_storage).Sum(schsDtemp);

    if(threadIdx.x==0 && threadIdx.y==0){
        atomicAdd2(achsDtemp_re, schsDtemp_red.real());
        atomicAdd2(achsDtemp_im, schsDtemp_red.imag());
    }
}
#endif
