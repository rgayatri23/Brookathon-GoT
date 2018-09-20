//#include "../ComplexClass/GPUComplex.h"
#include "types.h"
#include "../ComplexClass/cudaAlloc.h"
#include "cub/cub.cuh"
#include "kernels.cuh"

/* #define __Kernel_2D */

__device__ void d_compute_fact(REAL wx, int nFreq, REAL *dFreqGrid, REAL &fact1, REAL &fact2, int &ifreq, int loop, bool flag_occ)
{
    if(loop == 1){
        REAL awx = abs(wx);
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

__device__ void d_ssxDittt_kernel(int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, REAL *vcoul, GPUComplex *I_eps_array, GPUComplex &ssxDittt, int ngpown, int ncouls, int n1,int ifreq, REAL fact1, REAL fact2)
{
    __shared__ REAL ssxDittt_re, ssxDittt_im;
    ssxDittt_re = 0.00; ssxDittt_im = 0.00;

    int loopOverngpown = 1, leftOverngpown = 0;
    if(blockDim.x < ngpown)
    {
        loopOverngpown = ngpown / blockDim.x;
        leftOverngpown = ngpown % blockDim.x;
    }

    REAL ssxDittt_re_loc = 0.00, ssxDittt_im_loc = 0.00;
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

__device__ void d_schDttt_corKernel1(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, REAL *vcoul, int ncouls, int ifreq, int ngpown, int n1, REAL fact1, REAL fact2)
{
    REAL schDttt_cor_re = 0.00, schDttt_cor_im = 0.00, \
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
            REAL schDttt_cor_re_loc = 0.00, schDttt_cor_im_loc = 0.00;
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
            REAL schDttt_cor_re_loc = 0.00, schDttt_cor_im_loc = 0.00;
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


__device__ void d_schDttt_corKernel2(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, REAL *vcoul, int ncouls, int ifreq, int ngpown, int n1, REAL fact1, REAL fact2, int iw)
{
    __shared__ REAL schDttt_cor_re , schDttt_cor_im ;
    schDttt_cor_re = 0.00; schDttt_cor_im = 0.00;
    int loopOverngpown = 1, leftOverngpown = 0;
    if(ngpown > blockDim.x)
    {
        loopOverngpown = ngpown / blockDim.x;
        leftOverngpown = ngpown % blockDim.x;
    }

    REAL schDttt_cor_re_loc = 0.00, schDttt_cor_im_loc = 0.00;
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

__global__ void asxDtemp_solver_2D(int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, REAL freqevalmin, REAL freqevalstep, REAL occ, REAL *ekq, REAL *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, REAL *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, REAL *asxDtemp_re, REAL *asxDtemp_im)
{
    const int n1 = blockIdx.y;
    const int iw = blockIdx.x;
    REAL wx = freqevalmin - ekq[n1] + freqevalstep;
    REAL fact1 = 0.00, fact2 = 0.00;
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

__global__ void asxDtemp_solver_1D(int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, REAL freqevalmin, REAL freqevalstep, REAL occ, REAL *ekq, REAL *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, REAL *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, REAL *asxDtemp_re, REAL *asxDtemp_im)
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
                REAL wx = freqevalmin - ekq[n1] + freqevalstep;
                REAL fact1 = 0.00, fact2 = 0.00;
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
                REAL wx = freqevalmin - ekq[n1] + freqevalstep;
                REAL fact1 = 0.00, fact2 = 0.00;
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


__global__ void achDtemp_cor_solver_2D(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, REAL freqevalmin, REAL freqevalstep, REAL *ekq, REAL *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, REAL *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *ach2Dtemp, REAL *achDtemp_cor_re, REAL *achDtemp_cor_im, GPUComplex *achDtemp_corb)
{
    const int n1 = blockIdx.y;
    const int iw = blockIdx.x;
    //    if(n1 < number_bands && iw < nfreqeval)
    {
        bool flag_occ;
        REAL fact1 = 0.00, fact2 = 0.00, wx = 0.00;
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

__global__ void achDtemp_cor_solver_1D(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, REAL freqevalmin, REAL freqevalstep, REAL *ekq, REAL *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, REAL *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *ach2Dtemp, REAL *achDtemp_cor_re, REAL *achDtemp_cor_im, GPUComplex *achDtemp_corb)
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
            REAL fact1, fact2, wx;
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
            REAL fact1, fact2, wx;
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


void d_achsDtemp_Kernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, REAL *vcoul, GPUComplex *achsDtemp)
{

#ifdef __Kernel_2D
#warning "Using 2D kernels"
    const int ntx=8, nty=128;
    dim3 numThreads(ntx, nty);
    dim3 numBlocks( (ncouls+numThreads.x-1)/numThreads.x, (ncouls+numThreads.x-1)/numThreads.x );

    achsDtemp_solver_2D<ntx, nty><<<numBlocks, numThreads>>>(number_bands, ngpown, ncouls, inv_igp_index, indinv, aqsntemp, aqsmtemp, I_epsR_array, vcoul, achsDtemp_re, achsDtemp_im);
    gpuErrchk(cudaPeekAtLastError());
#else
#warning "Using 1D kernels"
    dim3 numBlocks(number_bands, 1, 1);
    const int numThreadsPerBlock=256;

    /* const size_t totalThreads = numBlocks.x*numThreadsPerBlock;    */
    GPUComplex *achsDtemp_reduce;
    
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    
    // Get reduction size
    gpuErrchk( cudaMalloc(&achsDtemp_reduce, numBlocks.x*sizeof(GPUComplex)) );
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, achsDtemp_reduce, achsDtemp, numBlocks.x);
    gpuErrchk( cudaMalloc(&d_temp_storage, temp_storage_bytes) );
    

    achsDtemp_solver_1D<numThreadsPerBlock><<<numBlocks, numThreadsPerBlock>>>(number_bands, ngpown, ncouls, inv_igp_index, indinv, aqsntemp, aqsmtemp, I_epsR_array, vcoul, achsDtemp_reduce);
    
    // Reduce
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, achsDtemp_reduce, achsDtemp, numBlocks.x);

    gpuErrchk( cudaFree(achsDtemp_reduce) );
    cudaFree(d_temp_storage);
#endif
}

void d_achsDtemp_mixed_Kernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, REAL *vcoul, double *achsDtemp_re, double *achsDtemp_im)
{
    dim3 numBlocks(number_bands, 1, 1);
    const int numThreadsPerBlock=128;
    achsDtemp_solver_1D_mixed<numThreadsPerBlock><<<numBlocks, numThreadsPerBlock>>>(number_bands, ngpown, ncouls, inv_igp_index, indinv, aqsntemp, aqsmtemp, I_epsR_array, vcoul, achsDtemp_re, achsDtemp_im);
}

void d_asxDtemp_Kernel(int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, REAL freqevalmin, REAL freqevalstep, REAL occ, REAL *ekq, REAL *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, REAL *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, REAL *asxDtemp_re, REAL *asxDtemp_im)
{
#ifdef __Kernel_2D
    dim3 numBlocks(nfreqeval, nvband);
    int numThreadsPerBlock=8;

    asxDtemp_solver_2D<<<numBlocks, numThreadsPerBlock>>>(nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, occ, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, asxDtemp_re, asxDtemp_im);
#else
    dim3 numBlocks(nvband, 1, 1);
    int numThreadsPerBlock=8;
    asxDtemp_solver_1D<<<numBlocks, numThreadsPerBlock>>>(nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, occ, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, asxDtemp_re, asxDtemp_im);
#endif

}

void d_achDtemp_cor_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, REAL freqevalmin, REAL freqevalstep, REAL *ekq, REAL *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, REAL *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *ach2Dtemp, REAL *achDtemp_cor_re, REAL *achDtemp_cor_im, GPUComplex *achDtemp_corb)
{
#ifdef __Kernel_2D
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
