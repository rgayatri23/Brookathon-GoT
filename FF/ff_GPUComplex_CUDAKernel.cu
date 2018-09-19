//#include "../ComplexClass/GPUComplex.h"
#include <thrust/complex.h>
#include "../ComplexClass/cudaAlloc.h"
#include "cub/cub.cuh"
#include "kernels.cuh"

using GPUComplex = thrust::complex<double>;

#define __Kernel_2D 0

__device__ void d_compute_fact(double wx, int nFreq, double *dFreqGrid, double &fact1, double &fact2, int &ifreq, int loop, bool flag_occ)
{
    if(loop == 1){
        double awx = abs(wx);
        for(int ijk = 0; ijk < nFreq-1; ++ijk)
        {
            if(awx > dFreqGrid[ijk] && awx < dFreqGrid[ijk+1])
                ifreq = ijk;
        }
        if(ifreq == 0) ifreq = nFreq-2;
        fact1 = (dFreqGrid[ifreq+1] - awx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
        fact2 = (awx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
    }
    if(loop == 2 && wx > 0.00)
    {
        for(int ijk = 0; ijk < nFreq-1; ++ijk)
        {
            if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                ifreq = ijk;
        }
        if(ifreq == -1) ifreq = nFreq-2;
        fact1 = -0.5 * (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
        fact2 = -0.5 * (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
    }
    else if(loop == 2 && flag_occ)
    {
        wx = -wx; ifreq = 0;
        for(int ijk = 0; ijk < nFreq-1; ++ijk)
        {
            if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                ifreq = ijk;
        }
        if(ifreq == 0) ifreq = nFreq-2;
        fact1 = (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
        fact2 = (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);

    }
}

__device__ void d_ssxDittt_kernel(int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_eps_array, GPUComplex &ssxDittt, int ngpown, int ncouls, int n1,int ifreq, double fact1, double fact2)
{
    __shared__ double ssxDittt_re, ssxDittt_im;
    ssxDittt_re = 0.00; ssxDittt_im = 0.00;

    int loopOverngpown = 1, leftOverngpown = 0;
    if(blockDim.x < ngpown)
    {
        loopOverngpown = ngpown / blockDim.x;
        leftOverngpown = ngpown % blockDim.x;
    }

    double ssxDittt_re_loc = 0.00, ssxDittt_im_loc = 0.00;
    for( int x = 0; x < loopOverngpown && threadIdx.x < blockDim.x ; ++x)
    {
        const int my_igp = x*blockDim.x + threadIdx.x;
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];
        GPUComplex ssxDitt(0.00, 0.00);

        for(int ig = 0; ig < ncouls; ++ig)
        {
            GPUComplex ssxDit = I_eps_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] * fact1 + \
                                I_eps_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] * fact2;

            ssxDitt += aqsntemp[n1*ncouls + ig] * thrust::conj(aqsmtemp[n1*ncouls + igp]) * ssxDit * vcoul[igp];
        }
        ssxDittt_re_loc += ssxDitt.real();
        ssxDittt_im_loc += ssxDitt.imag();
    }
    if(leftOverngpown)
    {
        const int my_igp = loopOverngpown*blockDim.x + threadIdx.x;
        if (my_igp < ngpown)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            GPUComplex ssxDitt(0.00, 0.00);

            for(int ig = 0; ig < ncouls; ++ig)
            {
                GPUComplex ssxDit = I_eps_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] * fact1 + \
                                    I_eps_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] * fact2;

                ssxDitt += aqsntemp[n1*ncouls + ig] * thrust::conj(aqsmtemp[n1*ncouls + igp]) * ssxDit * vcoul[igp];
            }
            ssxDittt_re_loc += ssxDitt.real();
            ssxDittt_im_loc += ssxDitt.imag();
        }
    }
    atomicAdd2(&ssxDittt_re, ssxDittt_re_loc);
    atomicAdd2(&ssxDittt_im, ssxDittt_im_loc);
    ssxDittt = GPUComplex (ssxDittt_re, ssxDittt_im);
}

__device__ void d_schDttt_corKernel1(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2)
{
    double schDttt_cor_re = 0.00, schDttt_cor_im = 0.00, \
                            schDttt_re = 0.00, schDttt_im = 0.00;

    int loopOverngpown = 1, leftOverngpown = 0;
    if(blockDim.x < ngpown)
    {
        loopOverngpown = ngpown / blockDim.x;
        leftOverngpown = ngpown % blockDim.x;
    }

    for( int x = 0; x < loopOverngpown && threadIdx.x < blockDim.x ; ++x)
    {
        int my_igp = x*blockDim.x + threadIdx.x;
        if (my_igp < ngpown)
        {
            double schDttt_cor_re_loc = 0.00, schDttt_cor_im_loc = 0.00;
            int indigp = inv_igp_index[my_igp] ;
            int igp = indinv[indigp];
            for(int ig = 0; ig < ncouls; ++ig)
            {
                GPUComplex sch2Dt = (I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig]) * fact1 + \
                                    (I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig]) * fact2;
                GPUComplex sch2Dtt = aqsntemp[n1*ncouls + ig] * thrust::conj(aqsmtemp[n1*ncouls + igp]) * sch2Dt * vcoul[igp];


                schDttt_re += sch2Dtt.real() ;
                schDttt_im += sch2Dtt.imag() ;
                schDttt_cor_re_loc += sch2Dtt.real() ;
                schDttt_cor_im_loc += sch2Dtt.imag() ;
            }
            atomicAdd2(&schDttt_cor_re, schDttt_cor_re_loc);
            atomicAdd2(&schDttt_cor_im, schDttt_cor_im_loc);
        }
    }
    if(leftOverngpown)
    {
        int my_igp = loopOverngpown*blockDim.x + threadIdx.x;
        if (my_igp < ngpown)
        {
            double schDttt_cor_re_loc = 0.00, schDttt_cor_im_loc = 0.00;
            int indigp = inv_igp_index[my_igp] ;
            int igp = indinv[indigp];
            for(int ig = 0; ig < ncouls; ++ig)
            {
                GPUComplex sch2Dt = (I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig]) * fact1 + \
                                    (I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig]) * fact2;
                GPUComplex sch2Dtt = aqsntemp[n1*ncouls + ig] * thrust::conj(aqsmtemp[n1*ncouls + igp]) * sch2Dt * vcoul[igp];


                schDttt_re += sch2Dtt.real() ;
                schDttt_im += sch2Dtt.imag() ;
                schDttt_cor_re += sch2Dtt.real() ;
                schDttt_cor_im += sch2Dtt.imag() ;
            }
            atomicAdd2(&schDttt_cor_re, schDttt_cor_re_loc);
            atomicAdd2(&schDttt_cor_im, schDttt_cor_im_loc);
        }
    }

    schDttt_cor = GPUComplex (schDttt_cor_re, schDttt_cor_im);
}


__device__ void d_schDttt_corKernel2(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2, int iw)
{
    __shared__ double schDttt_cor_re , schDttt_cor_im ;
    schDttt_cor_re = 0.00; schDttt_cor_im = 0.00;
    int loopOverngpown = 1, leftOverngpown = 0;
    if(ngpown > blockDim.x)
    {
        loopOverngpown = ngpown / blockDim.x;
        leftOverngpown = ngpown % blockDim.x;
    }

    double schDttt_cor_re_loc = 0.00, schDttt_cor_im_loc = 0.00;
    for(int x = 0; x < loopOverngpown && threadIdx.x < blockDim.x; ++x)
    {
        const int my_igp = x*blockDim.x + threadIdx.x;
        if(my_igp < ngpown)
        {
            int indigp = inv_igp_index[my_igp] ;
            int igp = indinv[indigp];
            for(int ig = 0; ig < ncouls; ++ig)
            {
                GPUComplex sch2Dt = ((I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[ifreq*ncouls*ngpown + my_igp*ncouls + ig]) * fact1 + \
                        (I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig]) * fact2) * -0.5;
                GPUComplex sch2Dtt = aqsntemp[n1*ncouls + ig] * thrust::conj(aqsmtemp[n1*ncouls + igp]) * sch2Dt * vcoul[igp];
                schDttt_cor_re_loc += sch2Dtt.real() ;
                schDttt_cor_im_loc += sch2Dtt.imag() ;
            }

        }
    }
    if(leftOverngpown)
    {
        const int my_igp = loopOverngpown*blockDim.x + threadIdx.x;
        if(my_igp < ngpown)
        {
            int indigp = inv_igp_index[my_igp] ;
            int igp = indinv[indigp];
            for(int ig = 0; ig < ncouls; ++ig)
            {
                GPUComplex sch2Dt = ((I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[ifreq*ncouls*ngpown + my_igp*ncouls + ig]) * fact1 + \
                        (I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig]) * fact2) * -0.5;
                GPUComplex sch2Dtt = aqsntemp[n1*ncouls + ig] * thrust::conj(aqsmtemp[n1*ncouls + igp]) * sch2Dt * vcoul[igp];
                schDttt_cor_re_loc += sch2Dtt.real() ;
                schDttt_cor_im_loc += sch2Dtt.imag() ;
            }

        }
    }
    atomicAdd2(&schDttt_cor_re , schDttt_cor_re_loc);
    atomicAdd2(&schDttt_cor_im , schDttt_cor_im_loc);
    schDttt_cor = GPUComplex (schDttt_cor_re, schDttt_cor_im);
}

//__global__ void achsDtemp_solver_2D(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, double *vcoul, double *achsDtemp_re, double *achsDtemp_im)
//{
//    //prepare redcution
//    typedef cub::BlockReduce<double, 32> BlockReduce;
//    
//    GPUComplex schsDtemp(0.00, 0.00);
//    
//    const int in = blockIdx.y * blockDim.y + threadIdx.y;
//    const int ig = blockIdx.x * blockDim.x + threadIdx.x;
//    //__shared__ double schsDtemp_block_re, schsDtemp_block_im;
//    double schsDtemp_red_re, schsDtemp_red_im;
//    __shared__ typename BlockReduce::TempStorage temp_storage;
//    
//    //schsDtemp_block_re = 0.;
//    //schsDtemp_block_im = 0.;
//    //__syncthreads();
//    
//    if( in < ngpown*number_bands && ig < ncouls ){
//        //extract indices
//        int my_igp = in % ngpown;
//        int n1 = (in - my_igp) / ngpown;
//        //do indirect access
//        int indigp = inv_igp_index[my_igp];
//        int igp = indinv[indigp];
//        
//        schsDtemp = schsDtemp - aqsntemp[n1*ncouls + ig] * thrust::conj(aqsmtemp[n1*ncouls + igp]) * I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig]* vcoul[ig] * 0.5;
//        
//        schsDtemp_red_re = BlockReduce(temp_storage).Sum(GPUComplex::real(schsDtemp));
//        schsDtemp_red_im = BlockReduce(temp_storage).Sum(GPUComplex::imag(schsDtemp));
//        //atomicAdd2(&schsDtemp_block_re, GPUComplex::real(schsDtemp));
//        //atomicAdd2(&schsDtemp_block_im, GPUComplex::imag(schsDtemp));
//        //__syncthreads();
//        //
//        if(threadIdx.x==0 && threadIdx.y==0){
//            //atomicAdd2(achsDtemp_re, schsDtemp_block_re);
//            //atomicAdd2(achsDtemp_im, schsDtemp_block_im);
//            atomicAdd2(achsDtemp_re, schsDtemp_red_re);
//            atomicAdd2(achsDtemp_im, schsDtemp_red_im);
//        }
//    }
//}


__global__ void achsDtemp_solver_1D(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, double *vcoul, double *achsDtemp_re, double *achsDtemp_im)
{
    const int n1 = blockIdx.x;
    int numBlk = (ncouls+blockDim.x-1)/blockDim.x;

    GPUComplex schsDtemp(0.00, 0.00);

    for(int my_igp = 0; my_igp < ngpown; ++my_igp)
    {
        int indigp = inv_igp_index[my_igp];          
        int igp = indinv[indigp];

        for(int blk = 0; blk < numBlk; blk++)
        {
            int ig = blk * blockDim.x + threadIdx.x;
            if (ig< ncouls)
                schsDtemp = schsDtemp - aqsntemp[n1*ncouls + ig] * thrust::conj(aqsmtemp[n1*ncouls + igp]) * I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig]* vcoul[ig] * 0.5;
        }
    }

    atomicAdd2(achsDtemp_re, schsDtemp.real());
    atomicAdd2(achsDtemp_im, schsDtemp.imag());
}

__global__ void asxDtemp_solver_2D(int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double occ, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, double *asxDtemp_re, double *asxDtemp_im)
{
    const int n1 = blockIdx.y;
    const int iw = blockIdx.x;
    double wx = freqevalmin - ekq[n1] + freqevalstep;
    double fact1 = 0.00, fact2 = 0.00;
    int ifreq = 0;
    GPUComplex ssxDittt(0.00, 0.00);

    d_compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 1, 0);

    if(wx > 0)
        d_ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2);
    else
        d_ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsA_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2);

    if(threadIdx.x == 0)
    {
        atomicAdd2(&asxDtemp_re[iw], (ssxDittt * occ).real());
        atomicAdd2(&asxDtemp_im[iw], (ssxDittt * occ).imag());
    }
}

__global__ void asxDtemp_solver_1D(int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double occ, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, double *asxDtemp_re, double *asxDtemp_im)
{
    const int n1 = blockIdx.x;
    //    if(n1 < nvband)
    {
        int loopOvernfreqeval = 1, leftOvernfreqeval = 0;
        if(nfreqeval > blockDim.x)
        {
            loopOvernfreqeval = nfreqeval / blockDim.x;
            leftOvernfreqeval = nfreqeval % blockDim.x;
        }
        for(int x = 0; x < loopOvernfreqeval && threadIdx.x < blockDim.x; ++x)
        {
            const int iw = x*blockDim.x + threadIdx.x;
            if(iw < nfreqeval)
            {
                double wx = freqevalmin - ekq[n1] + freqevalstep;
                double fact1 = 0.00, fact2 = 0.00;
                int ifreq = 0;
                GPUComplex ssxDittt(0.00, 0.00);

                d_compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 1, 0);

                if(wx > 0)
                    d_ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2);
                else
                    d_ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsA_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2);
                atomicAdd2(&asxDtemp_re[iw], (ssxDittt * occ).real());
                atomicAdd2(&asxDtemp_im[iw], (ssxDittt * occ).imag());
            }
        }
        if(leftOvernfreqeval)
        {
            const int iw = loopOvernfreqeval*blockDim.x + threadIdx.x;
            if(iw < nfreqeval)
            {
                double wx = freqevalmin - ekq[n1] + freqevalstep;
                double fact1 = 0.00, fact2 = 0.00;
                int ifreq = 0;
                GPUComplex ssxDittt(0.00, 0.00);

                d_compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 1, 0);

                if(wx > 0)
                    d_ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2);
                else
                    d_ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsA_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2);
                atomicAdd2(&asxDtemp_re[iw], (ssxDittt * occ).real());
                atomicAdd2(&asxDtemp_im[iw], (ssxDittt * occ).imag());
            }
        }
    }
}


__global__ void achDtemp_cor_solver_2D(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *ach2Dtemp, double *achDtemp_cor_re, double *achDtemp_cor_im, GPUComplex *achDtemp_corb)
{
    const int n1 = blockIdx.y;
    const int iw = blockIdx.x;
    //    if(n1 < number_bands && iw < nfreqeval)
    {
        bool flag_occ;
        double fact1 = 0.00, fact2 = 0.00, wx = 0.00;
        GPUComplex schDi_cor(0.00, 0.00), schDi_corb(0.00, 0.00);
        int ifreq = 0.00;
        flag_occ = n1 < nvband;
        wx = freqevalmin - ekq[n1] + freqevalstep;
        d_compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 2, flag_occ);

        if(wx > 0)
        {
            if(!flag_occ)
                d_schDttt_corKernel1(schDi_cor, inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2);
        }
        else if(flag_occ)
        {
            d_schDttt_corKernel2(schDi_cor, inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2, iw);
        }


        //Summing up at the end of iw loop by just the master thread
        if(threadIdx.x == 0)
        {
            atomicAdd2(&achDtemp_cor_re[iw], schDi_cor.real());
            atomicAdd2(&achDtemp_cor_im[iw], schDi_cor.imag());
        }
    } //n1
}

__global__ void achDtemp_cor_solver_1D(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *ach2Dtemp, double *achDtemp_cor_re, double *achDtemp_cor_im, GPUComplex *achDtemp_corb)
{
    const int n1 = blockIdx.x;
    //    if(n1 < number_bands)
    {
        int loopOvernfreqeval = 1, leftOvernfreqeval = 0;
        if(nfreqeval > blockDim.x)
        {
            loopOvernfreqeval = nfreqeval / blockDim.x;
            leftOvernfreqeval = nfreqeval % blockDim.x;
        }
        for(int x = 0; x < loopOvernfreqeval && threadIdx.x < blockDim.x; ++x)
        {
            bool flag_occ;
            const int iw = x*blockDim.x + threadIdx.x;
            double fact1, fact2, wx;
            GPUComplex schDi_cor(0.00, 0.00), schDi_corb(0.00, 0.00);
            int ifreq;
            flag_occ = n1 < nvband;
            wx = freqevalmin - ekq[n1] + freqevalstep;
            fact1 = 0.00, fact2 = 0.00;
            ifreq = 0.00;

            d_compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 2, flag_occ);


            if(wx > 0)
            {
                if(!flag_occ)
                    d_schDttt_corKernel1(schDi_cor, inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2);
            }
            else if(flag_occ)
                d_schDttt_corKernel2(schDi_cor, inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2, iw);

            atomicAdd2(&achDtemp_cor_re[iw], schDi_cor.real());
            atomicAdd2(&achDtemp_cor_im[iw], schDi_cor.imag());

        }
        if(leftOvernfreqeval)
        {
            bool flag_occ;
            const int iw = loopOvernfreqeval*blockDim.x + threadIdx.x;
            double fact1, fact2, wx;
            GPUComplex schDi_cor(0.00, 0.00), schDi_corb(0.00, 0.00);
            int ifreq;
            flag_occ = n1 < nvband;
            wx = freqevalmin - ekq[n1] + freqevalstep;
            fact1 = 0.00, fact2 = 0.00;
            ifreq = 0.00;

            d_compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 2, flag_occ);


            if(wx > 0)
            {
                if(!flag_occ)
                    d_schDttt_corKernel1(schDi_cor, inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2);
            }
            else if(flag_occ)
                d_schDttt_corKernel2(schDi_cor, inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2, iw);

            atomicAdd2(&achDtemp_cor_re[iw], schDi_cor.real());
            atomicAdd2(&achDtemp_cor_im[iw], schDi_cor.imag());

        }

    }
}


void d_achsDtemp_Kernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, double *vcoul, double *achsDtemp_re, double *achsDtemp_im)
{


#if defined(__Kernel_2D)
#warning "using 2D kernels"
    const int ntx=8, nty=128;
    dim3 numThreads(ntx, nty);
    dim3 numBlocks( int(ceil(size_t(ncouls)/double(numThreads.x))), int(ceil(size_t(number_bands)/double(numThreads.y))) );

    achsDtemp_solver_2D_v2<ntx, nty><<<numBlocks, numThreads>>>(number_bands, ngpown, ncouls, inv_igp_index, indinv, aqsntemp, aqsmtemp, I_epsR_array, vcoul, achsDtemp_re, achsDtemp_im);
    gpuErrchk(cudaPeekAtLastError());
#else
#warning "using 1D kernels"
    dim3 numBlocks(number_bands, 1, 1);
    int numThreadsPerBlock=16;
    achsDtemp_solver_1D<<<numBlocks, numThreadsPerBlock>>>(number_bands, ngpown, ncouls, inv_igp_index, indinv, aqsntemp, aqsmtemp, I_epsR_array, vcoul, achsDtemp_re, achsDtemp_im);
#endif
}

void d_asxDtemp_Kernel(int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double occ, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, double *asxDtemp_re, double *asxDtemp_im)
{
#if __Kernel_2D
    //    dim3 numBlocks(nvband, nfreqeval);
    dim3 numBlocks(nfreqeval, nvband);
    int numThreadsPerBlock=8;

    asxDtemp_solver_2D<<<numBlocks, numThreadsPerBlock>>>(nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, occ, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, asxDtemp_re, asxDtemp_im);
#else
    dim3 numBlocks(nvband, 1, 1);
    int numThreadsPerBlock=8;
    asxDtemp_solver_1D<<<numBlocks, numThreadsPerBlock>>>(nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, occ, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, asxDtemp_re, asxDtemp_im);
#endif

}

void d_achDtemp_cor_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *ach2Dtemp, double *achDtemp_cor_re, double *achDtemp_cor_im, GPUComplex *achDtemp_corb)
{
#if __Kernel_2D
    //    dim3 numBlocks(number_bands, nfreqeval);
    dim3 numBlocks(nfreqeval, number_bands);
    int numThreadsPerBlock=8;

    achDtemp_cor_solver_2D<<<numBlocks, numThreadsPerBlock>>>(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, ach2Dtemp, achDtemp_cor_re, achDtemp_cor_im, achDtemp_corb);
#else
    dim3 numBlocks(number_bands, 1, 1);
    int numThreadsPerBlock=32;

    achDtemp_cor_solver_1D<<<numBlocks, numThreadsPerBlock>>>(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, ach2Dtemp, achDtemp_cor_re, achDtemp_cor_im, achDtemp_corb);
#endif
}
