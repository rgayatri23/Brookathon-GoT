//#include "../ComplexClass/GPUComplex.h"
#include "types.h"
#include "../ComplexClass/cudaAlloc.h"
#include "kernels.h"

#define CUDA_VER 1

double elapsedTime(timeval start_time, timeval end_time)
{
    return ((end_time.tv_sec - start_time.tv_sec) +1e-6*(end_time.tv_usec - start_time.tv_usec));
}

void calculate_schDt_lin3(GPUComplex& schDt_lin3, GPUComplex* sch2Di, bool flag_occ, int freqevalmin, REAL *ekq, int iw, int freqevalstep, REAL cedifft_zb_right, REAL cedifft_zb_left, GPUComplex schDt_left, GPUComplex schDt_lin2, int n1, REAL pref_zb, GPUComplex pref_zb_compl, GPUComplex schDt_avg)
{
    REAL intfact = (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_left);
    if(intfact < 0.0001) intfact = 0.0001;
    if(intfact > 10000) intfact = 10000;
    intfact = -log(intfact);
    sch2Di[iw] = sch2Di[iw] - pref_zb_compl * schDt_avg * intfact;
    if(flag_occ)
    {
       REAL  intfact = abs((freqevalmin - ekq[n1] + (iw-1)*freqevalstep + cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1)*freqevalstep + cedifft_zb_left));
        if(intfact < 0.0001) intfact = 0.0001;
        if(intfact > 10000) intfact = 10000;
        intfact = log(intfact);
        schDt_lin3 = (schDt_left + schDt_lin2) * (-freqevalmin - ekq[n1] + (iw-1)*freqevalstep - cedifft_zb_left)*intfact ;
    }
    else
        schDt_lin3 = (schDt_left + schDt_lin2) * (freqevalmin - ekq[n1] + (iw-1)*freqevalstep - cedifft_zb_left)*intfact;

}

inline void compute_fact(REAL wx, int nFreq, REAL *dFreqGrid, REAL &fact1, REAL &fact2, int &ifreq, int loop, bool flag_occ)
{
    if(loop == 1 && wx > 0.00)
    {
            for(int ijk = 0; ijk < nFreq-1; ++ijk)
            {
                if(wx > dFreqGrid[ijk] && wx < dFreqGrid[ijk+1])
                ifreq = ijk;
            }
            if(ifreq == 0) ifreq = nFreq-2;
            fact1 = (dFreqGrid[ifreq+1] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
            fact2 = (wx - dFreqGrid[ifreq]) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
    }
    else if(loop == 1)
    {
        for(int ijk = 0; ijk < nFreq-1; ++ijk)
        {
            if(-wx > dFreqGrid[ijk] && -wx < dFreqGrid[ijk+1])
                ifreq = ijk;
        }
        if(ifreq == 0) ifreq = nFreq-2;
        fact1 = (dFreqGrid[ifreq+1] + wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
        fact2 = (-dFreqGrid[ifreq] - wx) / (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]);
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
//
inline void ssxDittt_kernel(int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, REAL *vcoul, GPUComplex *I_eps_array, GPUComplex &ssxDittt, int ngpown, int ncouls, int n1,int ifreq, REAL fact1, REAL fact2)
{
    REAL ssxDittt_re = 0.00, ssxDittt_im = 0.00;
    for(int my_igp = 0; my_igp < ngpown; ++my_igp)
    {
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];
        GPUComplex ssxDit(0.00, 0.00);
        GPUComplex ssxDitt(0.00, 0.00);

        for(int ig = 0; ig < ncouls; ++ig)
        {
            ssxDit = I_eps_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] * fact1 + \
                                         I_eps_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] * fact2;

            ssxDitt += aqsntemp[n1*ncouls + ig] * thrust::conj(aqsmtemp[n1*ncouls + igp]) * ssxDit * vcoul[igp];
        }
        ssxDittt_re += ssxDitt.real();
        ssxDittt_im += ssxDitt.imag();
    }
    ssxDittt = GPUComplex (ssxDittt_re, ssxDittt_im);
}


void achsDtemp_Kernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, REAL *vcoul, GPUComplex &achsDtemp)
{
    REAL achsDtemp_re = 0.00, achsDtemp_im = 0.00;
    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];

            GPUComplex schsDtemp(0.00, 0.00);

            for(int ig = 0; ig < ncouls; ++ig)
                schsDtemp = schsDtemp - aqsntemp[n1*ncouls + ig] * thrust::conj(aqsmtemp[n1*ncouls + igp]) * I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig]* vcoul[ig] * 0.5;

            achsDtemp_re += schsDtemp.real();
            achsDtemp_im += schsDtemp.imag();
        }
    } //n1
    achsDtemp = GPUComplex (achsDtemp_re, achsDtemp_im) ;

}
//
inline void asxDtemp_Kernel(int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, REAL freqevalmin, REAL freqevalstep, REAL occ, REAL *ekq, REAL *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, REAL *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *asxDtemp)
{
    GPUComplex expr0(0.00, 0.00);
    GPUComplex ssxDittt(0.00, 0.00);
    for(int n1 = 0; n1 < nvband; ++n1)
    {
        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            REAL wx = freqevalmin - ekq[n1] + freqevalstep;
            REAL fact1 = 0.00, fact2 = 0.00;
            int ifreq = 0;
            GPUComplex ssxDittt = expr0;

            compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 1, 0);

        if(wx > 0)
            ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2);
            else
                ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsA_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2);

            asxDtemp[iw] += ssxDittt * occ;
        } // iw
    }
}
//
void achDtemp_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, REAL freqevalmin, REAL freqevalstep, REAL *ekq, REAL pref_zb, REAL *pref, REAL *dFreqGrid, GPUComplex *dFreqBrd, GPUComplex *schDt_matrix, GPUComplex *schDi, GPUComplex *schDi_cor, GPUComplex *sch2Di, GPUComplex *achDtemp)
{
    bool flag_occ;
    GPUComplex expr0(0.00, 0.00);
    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        for(int ifreq = 0; ifreq < nFreq; ++ifreq)
        {
            flag_occ = n1 < nvband;
            GPUComplex schDt = schDt_matrix[n1*nFreq + ifreq];
            REAL cedifft_zb = dFreqGrid[ifreq];
            REAL cedifft_zb_right, cedifft_zb_left;
            GPUComplex schDt_right, schDt_left, schDt_avg, schDt_lin, schDt_lin2, schDt_lin3;
            GPUComplex cedifft_compl(cedifft_zb, 0.00);
            GPUComplex cedifft_cor;
            GPUComplex cedifft_coh = cedifft_compl - dFreqBrd[ifreq];
            GPUComplex pref_zb_compl(0.00, pref_zb);

            if(flag_occ)
                cedifft_cor = -cedifft_compl - dFreqBrd[ifreq];
                else
                    cedifft_cor = cedifft_compl - dFreqBrd[ifreq];

            if(ifreq != 0)
            {
                cedifft_zb_right = cedifft_zb;
                cedifft_zb_left = dFreqGrid[ifreq-1];
                schDt_right = schDt;
                schDt_left = schDt_matrix[n1*nFreq + ifreq-1];
                schDt_avg = (schDt_right + schDt_left) * 0.5;
                schDt_lin = schDt_right - schDt_left;
                schDt_lin2 = schDt_lin / (cedifft_zb_right - cedifft_zb_left);

                for(int iw = 0; iw < nfreqeval; ++iw)
                {
                    sch2Di[iw] = expr0;
                    calculate_schDt_lin3(schDt_lin3, sch2Di, flag_occ, freqevalmin, ekq, iw, freqevalstep, cedifft_zb_right, cedifft_zb_left, schDt_left, schDt_lin2, n1, pref_zb, pref_zb_compl, schDt_avg);

                    schDt_lin3 += schDt_lin;
                    schDi_cor[iw] = schDi_cor[iw] -  (pref_zb_compl * schDt_lin3);
                }
            }

            for(int iw = 0; iw < nfreqeval; ++iw)
            {
                schDi[iw] = expr0;
                REAL wx = freqevalmin - ekq[n1] + (iw-1) * freqevalstep;
                GPUComplex tmp(0.00, pref[ifreq]);
                schDi[iw] = schDi[iw] - ((tmp*schDt) / (wx- cedifft_coh));
                achDtemp[iw] += schDi[iw];
            }
        }
    }

}

inline void achDtemp_cor_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, REAL freqevalmin, REAL freqevalstep, REAL *ekq, REAL *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, REAL *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *schDi_cor, GPUComplex *schDi_corb, GPUComplex *sch2Di, GPUComplex *ach2Dtemp, GPUComplex *achDtemp_cor, GPUComplex *achDtemp_corb)
{
    bool flag_occ;
    GPUComplex expr0(0.00, 0.00);
    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        flag_occ = n1 < nvband;

        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            schDi_corb[iw] = expr0;
            schDi_cor[iw] = expr0;
            REAL wx = freqevalmin - ekq[n1] + freqevalstep;

            REAL fact1 = 0.00, fact2 = 0.00;
            int ifreq = 0.00;

            compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 2, flag_occ);

            if(wx > 0)
            {
                if(!flag_occ)
                schDttt_corKernel1(schDi_cor[iw], inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, sch2Di[iw],vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2);
            }
            else if(flag_occ)
                schDttt_corKernel2(schDi_cor[iw], inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2);


//Summing up at the end of iw loop
            ach2Dtemp[iw] += sch2Di[iw];
            achDtemp_cor[iw] += schDi_cor[iw];
            achDtemp_corb[iw] += schDi_corb[iw];
        }// iw
    } //n1
}

inline void schDttt_corKernel1(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex &schDttt, REAL *vcoul, int ncouls, int ifreq, int ngpown, int n1, REAL fact1, REAL fact2)
{
    int blkSize = 512;
    REAL schDttt_cor_re = 0.00, schDttt_cor_im = 0.00, \
        schDttt_re = 0.00, schDttt_im = 0.00;
    for(int igbeg = 0; igbeg < ncouls; igbeg += blkSize)
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            for(int ig = igbeg; ig < std::min(ncouls, igbeg+blkSize); ++ig)
            {
                int indigp = inv_igp_index[my_igp] ;
                int igp = indinv[indigp];
                GPUComplex sch2Dt = (I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig]) * fact1 + \
                                            (I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig]) * fact2;
                GPUComplex sch2Dtt = aqsntemp[n1*ncouls + ig] * thrust::conj(aqsmtemp[n1*ncouls + igp]) * sch2Dt * vcoul[igp];


                schDttt_re += sch2Dtt.real() ;
                schDttt_im += sch2Dtt.imag() ;
                schDttt_cor_re += sch2Dtt.real() ;
                schDttt_cor_im += sch2Dtt.imag() ;
            }
        }
    }
    schDttt_cor = GPUComplex (schDttt_cor_re, schDttt_cor_im);

}

inline void schDttt_corKernel2(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, REAL *vcoul, int ncouls, int ifreq, int ngpown, int n1, REAL fact1, REAL fact2)
{
    int blkSize = 512;
    REAL schDttt_cor_re = 0.00, schDttt_cor_im = 0.00;
    for(int igbeg = 0; igbeg < ncouls; igbeg += blkSize)
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            for(int ig = igbeg; ig < std::min(ncouls, igbeg+blkSize); ++ig)
            {
                int indigp = inv_igp_index[my_igp] ;
                int igp = indinv[indigp];
                GPUComplex sch2Dt = ((I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[ifreq*ncouls*ngpown + my_igp*ncouls + ig]) * fact1 + \
                                            (I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig]) * fact2) * -0.5;
                GPUComplex sch2Dtt = aqsntemp[n1*ncouls + ig] * thrust::conj(aqsmtemp[n1*ncouls + igp]) * sch2Dt * vcoul[igp];
                schDttt_cor_re += sch2Dtt.real() ;
                schDttt_cor_im += sch2Dtt.imag() ;
            }
        }
    }
    schDttt_cor = GPUComplex (schDttt_cor_re, schDttt_cor_im);
}

int main(int argc, char** argv)
{

    if(argc != 7)
    {
        std::cout << "Incorrect Parameters!!! The correct form is " << std::endl;
        std::cout << "./a.out number_bands nvband ncouls ngpown nFreq nfreqeval " << std::endl;
        exit(0);
    }

    auto startTimer = std::chrono::high_resolution_clock::now();
    int number_bands = atoi(argv[1]);
    int nvband = atoi(argv[2]);
    const int ncouls = atoi(argv[3]);
    const int ngpown = atoi(argv[4]);
    const int nFreq = atoi(argv[5]);
    const int nfreqeval = atoi(argv[6]);

    if(ngpown > ncouls)
    {
        std::cout << "Incorrect Parameters!!! ngpown cannot be greater than ncouls. The correct form is " << std::endl;
        std::cout << "./a.out number_bands nvband ncouls ngpown nFreq nfreqeval " << std::endl;
        exit(0);
    }

    timeval startTimer_Kernel, endTimer_Kernel, \
        start_achsDtemp_Kernel, end_achsDtemp_Kernel, \
        start_achsDtemp_mixed_Kernel, end_achsDtemp_mixed_Kernel, \
        start_asxDtemp_Kernel, end_asxDtemp_Kernel, \
        start_asxDtemp_mixed_Kernel, end_asxDtemp_mixed_Kernel, \
        start_achDtemp_Kernel, end_achDtemp_Kernel,
        start_achDtemp_cor_Kernel, end_achDtemp_cor_Kernel, \
        start_achDtemp_cor_mixed_Kernel, end_achDtemp_cor_mixed_Kernel, \
        start_preKernel, end_preKernel;

    gettimeofday(&start_preKernel, NULL);

#if CUDA_VER
    std::cout << "Cuda Version" << std::endl;
#else
    std::cout << "Seq Version" << std::endl;
#endif

    std::cout << "number_bands = " << number_bands << \
        "\n nvband = " << nvband << \
        "\n ncouls = " << ncouls << \
        "\n ngpown = " << ngpown << \
        "\n nFreq = " << nFreq << \
        "\n nfreqeval = " << nfreqeval << std::endl;

    GPUComplex expr0( 0.0 , 0.0);
    GPUComplex expr( 0.5 , 0.5);
    GPUComplex expR( 0.5 , 0.5);
    GPUComplex expA( 0.5 , -0.5);
    GPUComplex exprP1( 0.5 , 0.1);
    REAL pref_zb = 0.5 / 3.14;

//Start to allocate the data structures;
    // int *inv_igp_index = new int[ngpown];
    // int *indinv = new int[ncouls];
    // REAL *vcoul = new REAL[ncouls];
    // REAL *ekq = new REAL[number_bands];
    // REAL *dFreqGrid = new REAL[nFreq];
    REAL *pref = new REAL[nFreq];
    long double mem_alloc = 0.00;

    // GPUComplex *aqsntemp = new GPUComplex[number_bands * ncouls];
    // mem_alloc += (number_bands * ncouls * sizeof(GPUComplex));
    //
    // GPUComplex *aqsmtemp= new GPUComplex[number_bands * ncouls];
    // mem_alloc += (number_bands * ncouls * sizeof(GPUComplex));

    // GPUComplex *I_epsR_array = new GPUComplex[nFreq * ngpown * ncouls];
    // mem_alloc += (nFreq * ngpown * ncouls * sizeof(GPUComplex));
    //
    // GPUComplex *I_epsA_array = new GPUComplex[nFreq * ngpown * ncouls];
    // mem_alloc += (nFreq * ngpown * ncouls * sizeof(GPUComplex));

    // GPUComplex *schDi = new GPUComplex[nfreqeval];
    // GPUComplex *sch2Di = new GPUComplex[nfreqeval];
    // GPUComplex *schDi_cor = new GPUComplex[nfreqeval];
    // GPUComplex *schDi_corb = new GPUComplex[nfreqeval];
    // GPUComplex *achDtemp = new GPUComplex[nfreqeval];
    // GPUComplex *ach2Dtemp = new GPUComplex[nfreqeval];
    GPUComplex *achDtemp_cor = new GPUComplex[nfreqeval];
    // GPUComplex *achDtemp_corb = new GPUComplex[nfreqeval];
    GPUComplex *asxDtemp = new GPUComplex[nfreqeval];
    GPUComplex *dFreqBrd = new GPUComplex[nFreq];
    // mem_alloc += (nfreqeval * 9 * sizeof(GPUComplex));
    // mem_alloc += (nFreq * sizeof(GPUComplex)) ;
    //
    // GPUComplex *schDt_matrix = new GPUComplex[number_bands * nFreq];
    // mem_alloc += (nFreq * number_bands * sizeof(GPUComplex));


    //Variables used for Cuda:
    // REAL *asxDtemp_re = new REAL[nfreqeval];
    // REAL *asxDtemp_im = new REAL[nfreqeval];
    // REAL *achDtemp_cor_re = new REAL[nfreqeval];
    // REAL *achDtemp_cor_im = new REAL[nfreqeval];

#if CUDA_VER
    double *achsDtemp_reD = new double;
    double *achsDtemp_imD = new double;
    //Allocated Memory on Device
    GPUComplex *d_aqsmtemp, *d_aqsntemp, \
        *d_I_epsR_array, *d_I_epsA_array, *d_schDt_matrix, \
        *d_achsDtemp, *d_ach2Dtemp , *d_achDtemp_corb;

    int *d_inv_igp_index, *d_indinv;
    REAL *d_vcoul, *d_dFreqGrid, *d_achsDtemp_re, *d_achsDtemp_im, \
        *d_ekq, *d_asxDtemp_re, *d_asxDtemp_im, *d_achDtemp_cor_re, *d_achDtemp_cor_im;
    double *d_achsDtemp_reD, *d_achsDtemp_imD, *d_achsDtempD, \
        *d_asxDtemp_reD, *d_asxDtemp_imD, *d_achDtemp_cor_reD, *d_achDtemp_cor_imD ;

    //Input
    //real prec
    CudaSafeCall(cudaMallocManaged((void**) &d_aqsmtemp, number_bands*ncouls*sizeof(GPUComplex)));
    CudaSafeCall(cudaMallocManaged((void**) &d_aqsntemp, number_bands*ncouls*sizeof(GPUComplex)));
    CudaSafeCall(cudaMallocManaged((void**) &d_I_epsR_array, nFreq*ngpown*ncouls*sizeof(GPUComplex)));
    CudaSafeCall(cudaMallocManaged((void**) &d_I_epsA_array, nFreq*ngpown*ncouls*sizeof(GPUComplex)));
    CudaSafeCall(cudaMallocManaged((void**) &d_schDt_matrix, number_bands*nFreq*sizeof(GPUComplex)));
    CudaSafeCall(cudaMallocManaged((void**) &d_inv_igp_index, ngpown*sizeof(int)));
    CudaSafeCall(cudaMallocManaged((void**) &d_indinv, ncouls*sizeof(int)));
    CudaSafeCall(cudaMallocManaged((void**) &d_vcoul, ncouls*sizeof(REAL)));
    CudaSafeCall(cudaMallocManaged((void**) &d_dFreqGrid, nFreq*sizeof(REAL)));
    CudaSafeCall(cudaMallocManaged((void**) &d_ekq, number_bands*sizeof(REAL)));
    CudaSafeCall(cudaMallocManaged((void**) &d_ach2Dtemp, nFreq*sizeof(GPUComplex)));
    CudaSafeCall(cudaMallocManaged((void**) &d_achDtemp_cor_re, nfreqeval*sizeof(REAL)));
    CudaSafeCall(cudaMallocManaged((void**) &d_achDtemp_cor_im, nfreqeval*sizeof(REAL)));
    CudaSafeCall(cudaMallocManaged((void**) &d_achDtemp_corb, nfreqeval*sizeof(GPUComplex)));

    //Output
    //single prec
    CudaSafeCall(cudaMallocManaged((void**) &d_achsDtemp, sizeof(GPUComplex)));
    // CudaSafeCall(cudaMallocManaged((void**) &d_achsDtemp_re, sizeof(REAL)));
    // CudaSafeCall(cudaMallocManaged((void**) &d_achsDtemp_im, sizeof(REAL)));
    CudaSafeCall(cudaMallocManaged((void**) &d_achsDtemp_reD, sizeof(double)));
    CudaSafeCall(cudaMallocManaged((void**) &d_achsDtemp_imD, sizeof(double)));
    CudaSafeCall(cudaMallocManaged((void**) &d_asxDtemp_re, nfreqeval*sizeof(REAL)));
    CudaSafeCall(cudaMallocManaged((void**) &d_asxDtemp_im, nfreqeval*sizeof(REAL)));
    //double prec
    CudaSafeCall(cudaMallocManaged((void**) &d_achsDtempD, sizeof(GPUComplexD)));
    CudaSafeCall(cudaMallocManaged((void**) &d_achsDtemp_reD, sizeof(double)));
    CudaSafeCall(cudaMallocManaged((void**) &d_achsDtemp_imD, sizeof(double)));
    CudaSafeCall(cudaMallocManaged((void**) &d_asxDtemp_reD, nfreqeval*sizeof(double)));
    CudaSafeCall(cudaMallocManaged((void**) &d_asxDtemp_imD, nfreqeval*sizeof(double)));

#endif

    const REAL freqevalmin = 0.00;
    const REAL freqevalstep = 0.50;
    REAL dw = -10;
    REAL occ = 1.00;

    //Initialize the data structures
    for(int ig = 0; ig < ngpown; ++ig)
        d_inv_igp_index[ig] = ig;

    for(int ig = 0; ig < ncouls; ++ig)
        d_indinv[ig] = ig;

    for(int i=0; i<number_bands; ++i)
    {
        d_ekq[i] = dw;
        dw += 1.00;

        for(int j=0; j<ncouls; ++j)
        {
            d_aqsmtemp[i*ncouls+j] = expr;
            d_aqsntemp[i*ncouls+j] = expr;
        }

        for(int j=0; j<nFreq; ++j)
            d_schDt_matrix[i*nFreq + j] = expr0;
    }

    for(int i=0; i<ncouls; ++i)
        d_vcoul[i] = 1.00;

    for(int i=0; i<nFreq; ++i)
    {
        for(int j=0; j<ngpown; ++j)
        {
            for(int k=0; k<ncouls; ++k)
            {
                d_I_epsR_array[i*ngpown*ncouls + j * ncouls + k] = expR;
                d_I_epsA_array[i*ngpown*ncouls + j * ncouls + k] = expA;
            }
        }
    }

    dw = 0.00;
    for(int ijk = 0; ijk < nFreq; ++ijk)
    {
        dFreqBrd[ijk] = exprP1;
        d_dFreqGrid[ijk] = dw;
        dw += 2.00;
    }

    for(int ifreq = 0; ifreq < nFreq; ++ifreq)
    {
        if(ifreq < nFreq-1)
            pref[ifreq] = (d_dFreqGrid[ifreq+1] - d_dFreqGrid[ifreq]) / 3.14;
            else
                pref[ifreq] = pref[ifreq-1];

    }
    pref[0] *= 0.5; pref[nFreq-1] *= 0.5;

    for(int i = 0; i < nfreqeval; ++i)
    {
        // d_schDi[i] = expr0;
        // d_sch2Di[i] = expr0;
        // d_schDi_corb[i] = expr0;
        // d_schDi_cor[i] = expr0;
        // d_asxDtemp[i] = expr0;
        // d_achDtemp[i] = expr0;
        d_ach2Dtemp[i] = expr0;
        // d_achDtemp_cor[i] = expr0;
        d_achDtemp_corb[i] = expr0;
        d_achDtemp_cor_re[i] = 0.00;
        d_achDtemp_cor_im[i] = 0.00;
    }
    GPUComplex *ssxDittt = new GPUComplex;

#if CUDA_VER
    //Copy data-strudtures value from host to device
    CudaSafeCall(cudaMemPrefetchAsync(d_aqsmtemp, number_bands*ncouls*sizeof(GPUComplex), 0, 0));
    CudaSafeCall(cudaMemPrefetchAsync(d_aqsntemp, number_bands*ncouls*sizeof(GPUComplex), 0, 0));
    CudaSafeCall(cudaMemPrefetchAsync(d_I_epsR_array, nFreq*ngpown*ncouls*sizeof(GPUComplex), 0, 0));
    CudaSafeCall(cudaMemPrefetchAsync(d_I_epsA_array, nFreq*ngpown*ncouls*sizeof(GPUComplex), 0, 0));
    CudaSafeCall(cudaMemPrefetchAsync(d_schDt_matrix, number_bands*nFreq*sizeof(GPUComplex), 0, 0));
    CudaSafeCall(cudaMemPrefetchAsync(d_indinv, ncouls*sizeof(int), 0, 0));
    CudaSafeCall(cudaMemPrefetchAsync(d_inv_igp_index, ngpown*sizeof(int), 0, 0));
    CudaSafeCall(cudaMemPrefetchAsync(d_vcoul, ncouls*sizeof(REAL), 0, 0));
    CudaSafeCall(cudaMemPrefetchAsync(d_dFreqGrid, nFreq*sizeof(REAL), 0, 0));
    CudaSafeCall(cudaMemPrefetchAsync(d_ekq, number_bands*sizeof(REAL), 0, 0));
    CudaSafeCall(cudaMemPrefetchAsync(d_achDtemp_corb, nfreqeval*sizeof(GPUComplex), 0, 0));
    CudaSafeCall(cudaMemPrefetchAsync(d_achDtemp_cor_re, nfreqeval*sizeof(REAL), 0, 0));
    CudaSafeCall(cudaMemPrefetchAsync(d_achDtemp_cor_im, nfreqeval*sizeof(REAL), 0, 0));
    
    cudaDeviceSynchronize();
#endif

    gettimeofday(&end_preKernel, NULL);
    double elapsed_preKernel = elapsedTime(start_preKernel, end_preKernel);
    std::cout << "pre kernel time taken = " << elapsed_preKernel << " secs" << std::endl;

    std::cout << "starting Kernels" << std::endl;
    gettimeofday(&startTimer_Kernel, NULL);

    /***********achsDtemp Kernel ****************/
    gettimeofday(&start_achsDtemp_Kernel, NULL);
#if !CUDA_VER
    GPUComplex achsDtemp(0.00, 0.00);
    achsDtemp_Kernel(number_bands, ngpown, ncouls, inv_igp_index, indinv, aqsntemp, aqsmtemp, d_I_epsR_array, vcoul, achsDtemp);
#else
    d_achsDtemp_Kernel(number_bands, ngpown, ncouls, d_inv_igp_index, d_indinv, d_aqsntemp, d_aqsmtemp, d_I_epsR_array, d_vcoul, d_achsDtemp);
    cudaDeviceSynchronize();
#endif
    gettimeofday(&end_achsDtemp_Kernel, NULL);
    
    /***********achsDtemp_mixed Kernel ****************/
    gettimeofday(&start_achsDtemp_mixed_Kernel, NULL);
#if !CUDA_VER
    GPUComplexD achsDtempD(0.00, 0.00);
    achsDtemp_mixed_Kernel(number_bands, ngpown, ncouls, inv_igp_index, indinv, aqsntemp, aqsmtemp, d_I_epsR_array, vcoul, achsDtempD);
#else
    d_achsDtemp_mixed_Kernel(number_bands, ngpown, ncouls, d_inv_igp_index, d_indinv, d_aqsntemp, d_aqsmtemp, d_I_epsR_array, d_vcoul, d_achsDtemp_reD, d_achsDtemp_imD);
    cudaDeviceSynchronize();
    CudaSafeCall(cudaMemcpy(achsDtemp_reD, d_achsDtemp_reD, sizeof(double), cudaMemcpyDeviceToHost));
#endif
    gettimeofday(&end_achsDtemp_mixed_Kernel, NULL);

    /***********asxDtemp Kernel ****************/
    gettimeofday(&start_asxDtemp_Kernel, NULL);
#if !CUDA_VER
    asxDtemp_Kernel(nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, occ, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, d_I_epsR_array, d_I_epsA_array, asxDtemp);
#else
    d_asxDtemp_Kernel(nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, occ, d_ekq, d_dFreqGrid, d_inv_igp_index, d_indinv, d_aqsmtemp, d_aqsntemp, d_vcoul, d_I_epsR_array, d_I_epsA_array, d_asxDtemp_re, d_asxDtemp_im);
    cudaDeviceSynchronize();
#endif
    gettimeofday(&end_asxDtemp_Kernel, NULL);

//    /***********achDtemp Kernel ****************/
    // achDtemp_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, ekq, pref_zb, pref, dFreqGrid, dFreqBrd, schDt_matrix, schDi, schDi_cor, sch2Di, asxDtemp);

    /***********achDtemp_cor Kernel ****************/
    auto startTimer_achDtemp_cor = std::chrono::high_resolution_clock::now();
    gettimeofday(&start_achDtemp_cor_Kernel, NULL);
#if !CUDA_VER
    achDtemp_cor_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, d_I_epsR_array, d_I_epsA_array, schDi_cor, schDi_corb, sch2Di, ach2Dtemp, achDtemp_cor, achDtemp_corb);
#else
    d_achDtemp_cor_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, d_ekq, d_dFreqGrid, d_inv_igp_index, d_indinv, d_aqsmtemp, d_aqsntemp, d_vcoul, d_I_epsR_array, d_I_epsA_array, d_ach2Dtemp, d_achDtemp_cor_re, d_achDtemp_cor_im, d_achDtemp_corb);
    cudaDeviceSynchronize();
#endif
    gettimeofday(&end_achDtemp_cor_Kernel, NULL);

#if CUDA_VER
//    cudaDeviceSynchronize();
    //achsDtemp
    
    //achsDtemp_mixed
    CudaSafeCall(cudaMemcpy(achsDtemp_reD, d_achsDtemp_reD, sizeof(double), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(achsDtemp_imD, d_achsDtemp_imD, sizeof(double), cudaMemcpyDeviceToHost));
    GPUComplexD achsDtempD(*achsDtemp_reD, *achsDtemp_imD);

    //asxDtemp
    // CudaSafeCall(cudaMemcpy(asxDtemp_re, d_asxDtemp_re, nfreqeval*sizeof(REAL), cudaMemcpyDeviceToHost));
    // CudaSafeCall(cudaMemcpy(asxDtemp_im, d_asxDtemp_im, nfreqeval*sizeof(REAL), cudaMemcpyDeviceToHost));

    //achDtemp_cor
    // CudaSafeCall(cudaMemcpy(achDtemp_cor_re, d_achDtemp_cor_re, nfreqeval*sizeof(REAL), cudaMemcpyDeviceToHost));
    // CudaSafeCall(cudaMemcpy(achDtemp_cor_im, d_achDtemp_cor_im, nfreqeval*sizeof(REAL), cudaMemcpyDeviceToHost));

    for(int iw = 0; iw < nfreqeval; ++iw)
    {
        asxDtemp[iw] = GPUComplex(d_asxDtemp_re[iw], d_asxDtemp_im[iw]);
        achDtemp_cor[iw] = GPUComplex(d_achDtemp_cor_re[iw], d_achDtemp_cor_im[iw]);
    }
#endif

    gettimeofday(&endTimer_Kernel, NULL);
    double elapsed_achsDtemp = elapsedTime(start_achsDtemp_Kernel, end_achsDtemp_Kernel);
    double elapsed_achsDtemp_mixed = elapsedTime(start_achsDtemp_mixed_Kernel, end_achsDtemp_mixed_Kernel);
    double elapsed_asxDtemp = elapsedTime(start_asxDtemp_Kernel, end_asxDtemp_Kernel);
    double elapsed_achDtemp_cor = elapsedTime(start_achDtemp_cor_Kernel, end_achDtemp_cor_Kernel);
    double elapsedTimer_Kernel = elapsedTime(startTimer_Kernel, endTimer_Kernel);

    std::cout.precision(13);
    std::cout << "achsDtemp = (" << d_achsDtemp->real() << ", " << d_achsDtemp->imag() << ")" << std::endl;
    std::cout << "achsDtempD = (" << achsDtempD.real() << ", " << achsDtempD.imag() << ")" << std::endl;
    //achsDtemp.print();
    std::cout << "asxDtemp[0] = (" << asxDtemp[0].real() << ", " << asxDtemp[0].imag() << ")" << std::endl;
    //asxDtemp[0].print();
    std::cout << "achDtemp_cor[0] = (" << achDtemp_cor[0].real() << ", " << achDtemp_cor[0].imag() << ")" << std::endl;
    //achDtemp_cor[0].print();

    std::cout << "********** achsDtemp Time Taken **********= " << elapsed_achsDtemp << " secs" << std::endl;
    std::cout << "********** achsDtemp_mixed Time Taken **********= " << elapsed_achsDtemp_mixed << " secs" << std::endl;
    std::cout << "********** asxDtemp Time Taken **********= " << elapsed_asxDtemp << " secs" << std::endl;
    std::cout << "********** achDtemp_cor Time Taken **********= " << elapsed_achDtemp_cor << " secs" << std::endl;
    std::cout << "********** Kernel Time Taken **********= " << elapsedTimer_Kernel << " secs" << std::endl;

#if CUDA_VER
    //Free Device Memory
    cudaFree(d_aqsmtemp);
    cudaFree(d_aqsntemp);
    cudaFree(d_I_epsR_array);
    cudaFree(d_I_epsA_array);
    cudaFree(d_schDt_matrix);
    cudaFree(d_inv_igp_index);
    cudaFree(d_indinv);
    cudaFree(d_achsDtemp);
    cudaFree(d_achsDtemp_reD);
    cudaFree(d_achsDtemp_imD);
    cudaFree(d_asxDtemp_re);
    cudaFree(d_asxDtemp_im);
    cudaFree(d_achDtemp_cor_re);
    cudaFree(d_achDtemp_cor_im);
    
    //double prec
    cudaFree(d_achsDtempD);
    cudaFree(d_achsDtemp_reD);
    cudaFree(d_achsDtemp_imD);
    cudaFree(d_asxDtemp_reD);
    cudaFree(d_asxDtemp_imD);
    cudaFree(d_achDtemp_cor_reD);
    cudaFree(d_achDtemp_cor_imD);
#endif

//Free the allocated memory
    free(pref);
    free(achDtemp_cor);
    free(asxDtemp);
    free(dFreqBrd);

    return 0;
}
