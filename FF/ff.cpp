#include "../ComplexClass/CustomComplex.h"
#include "../ComplexClass/cudaAlloc.h"

#define CUDA_VER 1

double elapsedTime(timeval start_time, timeval end_time)
{
    return ((end_time.tv_sec - start_time.tv_sec) +1e-6*(end_time.tv_usec - start_time.tv_usec));
}

void calculate_schDt_lin3(CustomComplex<double>& schDt_lin3, CustomComplex<double>* sch2Di, bool flag_occ, int freqevalmin, double *ekq, int iw, int freqevalstep, double cedifft_zb_right, double cedifft_zb_left, CustomComplex<double> schDt_left, CustomComplex<double> schDt_lin2, int n1, double pref_zb, CustomComplex<double> pref_zb_compl, CustomComplex<double> schDt_avg)
{
    double intfact = (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_left);
    if(intfact < 0.0001) intfact = 0.0001;
    if(intfact > 10000) intfact = 10000;
    intfact = -log(intfact);
    sch2Di[iw] = sch2Di[iw] - pref_zb_compl * schDt_avg * intfact;
    if(flag_occ)
    {
       double  intfact = abs((freqevalmin - ekq[n1] + (iw-1)*freqevalstep + cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1)*freqevalstep + cedifft_zb_left));
        if(intfact < 0.0001) intfact = 0.0001;
        if(intfact > 10000) intfact = 10000;
        intfact = log(intfact);
        schDt_lin3 = (schDt_left + schDt_lin2) * (-freqevalmin - ekq[n1] + (iw-1)*freqevalstep - cedifft_zb_left)*intfact ;
    }
    else
        schDt_lin3 = (schDt_left + schDt_lin2) * (freqevalmin - ekq[n1] + (iw-1)*freqevalstep - cedifft_zb_left)*intfact;

}

inline void compute_fact(double wx, int nFreq, double *dFreqGrid, double &fact1, double &fact2, int &ifreq, int loop, bool flag_occ)
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

inline void ssxDittt_kernel(int *inv_igp_index, int *indinv, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, CustomComplex<double> *I_eps_array, CustomComplex<double> &ssxDittt, int ngpown, int ncouls, int n1,int ifreq, double fact1, double fact2)
{
    double ssxDittt_re = 0.00, ssxDittt_im = 0.00;
    for(int my_igp = 0; my_igp < ngpown; ++my_igp)
    {
        int indigp = inv_igp_index[my_igp];
        int igp = indinv[indigp];
        CustomComplex<double> ssxDit(0.00, 0.00);
        CustomComplex<double> ssxDitt(0.00, 0.00);

        for(int ig = 0; ig < ncouls; ++ig)
        {
            ssxDit = I_eps_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] * fact1 + \
                                         I_eps_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] * fact2;

            ssxDitt += aqsntemp[n1*ncouls + ig] * CustomComplex_conj(aqsmtemp[n1*ncouls + igp]) * ssxDit * vcoul[igp];
        }
        ssxDittt_re += CustomComplex_real(ssxDitt);
        ssxDittt_im += CustomComplex_imag(ssxDitt);
    }
    ssxDittt = CustomComplex<double> (ssxDittt_re, ssxDittt_im);
}


void achsDtemp_Kernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, CustomComplex<double> *aqsntemp, CustomComplex<double> *aqsmtemp, CustomComplex<double> *I_epsR_array, double *vcoul, CustomComplex<double> &achsDtemp)
{
    double achsDtemp_re = 0.00, achsDtemp_im = 0.00;
    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];

            CustomComplex<double> schsDtemp(0.00, 0.00);

            for(int ig = 0; ig < ncouls; ++ig)
                schsDtemp = schsDtemp - aqsntemp[n1*ncouls + ig] * CustomComplex_conj(aqsmtemp[n1*ncouls + igp]) * I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig]* vcoul[ig] * 0.5;

            achsDtemp_re += CustomComplex_real(schsDtemp);
            achsDtemp_im += CustomComplex_imag(schsDtemp);
        }
    } //n1
    achsDtemp = CustomComplex<double> (achsDtemp_re, achsDtemp_im) ;

}

inline void asxDtemp_Kernel(int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double occ, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *asxDtemp)
{
    CustomComplex<double> expr0(0.00, 0.00);
    CustomComplex<double> ssxDittt(0.00, 0.00);
    for(int n1 = 0; n1 < nvband; ++n1)
    {
        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            double wx = freqevalmin - ekq[n1] + freqevalstep;
            double fact1 = 0.00, fact2 = 0.00;
            int ifreq = 0;
            CustomComplex<double> ssxDittt = expr0;

            compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 1, 0);

        if(wx > 0)
            ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2);
            else
                ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsA_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2);

            asxDtemp[iw] += ssxDittt * occ;
        } // iw
    }
}

//void achDtemp_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double pref_zb, double *pref, double *dFreqGrid, CustomComplex<double> *dFreqBrd, CustomComplex<double> *schDt_matrix, CustomComplex<double> *schDi, CustomComplex<double> *schDi_cor, CustomComplex<double> *sch2Di, CustomComplex<double> *achDtemp)
//{
//    bool flag_occ;
//    CustomComplex<double> expr0(0.00, 0.00);
//    for(int n1 = 0; n1 < number_bands; ++n1)
//    {
//        for(int ifreq = 0; ifreq < nFreq; ++ifreq)
//        {
//            flag_occ = n1 < nvband;
//            CustomComplex<double> schDt = schDt_matrix[n1*nFreq + ifreq];
//            double cedifft_zb = dFreqGrid[ifreq];
//            double cedifft_zb_right, cedifft_zb_left;
//            CustomComplex<double> schDt_right, schDt_left, schDt_avg, schDt_lin, schDt_lin2, schDt_lin3;
//            CustomComplex<double> cedifft_compl(cedifft_zb, 0.00);
//            CustomComplex<double> cedifft_cor;
//            CustomComplex<double> cedifft_coh = cedifft_compl - dFreqBrd[ifreq];
//            CustomComplex<double> pref_zb_compl(0.00, pref_zb);
//
//            if(flag_occ)
//                cedifft_cor = cedifft_compl * -1 - dFreqBrd[ifreq];
//                else
//                    cedifft_cor = cedifft_compl - dFreqBrd[ifreq];
//
//            if(ifreq != 0)
//            {
//                cedifft_zb_right = cedifft_zb;
//                cedifft_zb_left = dFreqGrid[ifreq-1];
//                schDt_right = schDt;
//                schDt_left = schDt_matrix[n1*nFreq + ifreq-1];
//                schDt_avg = (schDt_right + schDt_left) * 0.5;
//                schDt_lin = schDt_right - schDt_left;
//                schDt_lin2 = schDt_lin / (cedifft_zb_right - cedifft_zb_left);
//
//                for(int iw = 0; iw < nfreqeval; ++iw)
//                {
//                    sch2Di[iw] = expr0;
//                    calculate_schDt_lin3(schDt_lin3, sch2Di, flag_occ, freqevalmin, ekq, iw, freqevalstep, cedifft_zb_right, cedifft_zb_left, schDt_left, schDt_lin2, n1, pref_zb, pref_zb_compl, schDt_avg);
//
//                    schDt_lin3 += schDt_lin;
//                    schDi_cor[iw] = schDi_cor[iw] -  (pref_zb_compl * schDt_lin3);
//                }
//            }
//
//            for(int iw = 0; iw < nfreqeval; ++iw)
//            {
//                schDi[iw] = expr0;
//                double wx = freqevalmin - ekq[n1] + (iw-1) * freqevalstep;
//                CustomComplex<double> tmp(0.00, pref[ifreq]);
//                schDi[iw] = schDi[iw] - ((tmp*schDt) / (wx- cedifft_coh));
//                achDtemp[iw] += schDi[iw];
//            }
//        }
//    }
//
//}

//inline void achDtemp_cor_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *schDi_cor, CustomComplex<double> *schDi_corb, CustomComplex<double> *sch2Di, CustomComplex<double> *ach2Dtemp, CustomComplex<double> *achDtemp_cor, CustomComplex<double> *achDtemp_corb)
//{
//    bool flag_occ;
//    CustomComplex<double> expr0(0.00, 0.00);
//    for(int n1 = 0; n1 < number_bands; ++n1)
//    {
//        flag_occ = n1 < nvband;
//
//        for(int iw = 0; iw < nfreqeval; ++iw)
//        {
//            schDi_corb[iw] = expr0;
//            schDi_cor[iw] = expr0;
//            double wx = freqevalmin - ekq[n1] + freqevalstep;
//
//            double fact1 = 0.00, fact2 = 0.00;
//            int ifreq = 0.00;
//
//            compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 2, flag_occ);
//
//            if(wx > 0)
//            {
//                if(!flag_occ)
//                schDttt_corKernel1(schDi_cor[iw], inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, sch2Di[iw],vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2);
//            }
//            else if(flag_occ)
//                schDttt_corKernel2(schDi_cor[iw], inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2);
//
//
////Summing up at the end of iw loop
//            ach2Dtemp[iw] += sch2Di[iw];
//            achDtemp_cor[iw] += schDi_cor[iw];
//            achDtemp_corb[iw] += schDi_corb[iw];
//        }// iw
//    } //n1
//}
//
////template<class T>
//inline void schDttt_corKernel1(CustomComplex<double> &schDttt_cor, int *inv_igp_index, int *indinv, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, CustomComplex<double> &schDttt, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2)
//{
//    int blkSize = 512;
//    double schDttt_cor_re = 0.00, schDttt_cor_im = 0.00, \
//        schDttt_re = 0.00, schDttt_im = 0.00;
//    for(int igbeg = 0; igbeg < ncouls; igbeg += blkSize)
//    {
//        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
//        {
//            for(int ig = igbeg; ig < min(ncouls, igbeg+blkSize); ++ig)
//            {
//                int indigp = inv_igp_index[my_igp] ;
//                int igp = indinv[indigp];
//                CustomComplex<double> sch2Dt = (I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig]) * fact1 + \
//                                            (I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig]) * fact2;
//                CustomComplex<double> sch2Dtt = aqsntemp[n1*ncouls + ig] * CustomComplex_conj(aqsmtemp[n1*ncouls + igp]) * sch2Dt * vcoul[igp];
//
//
//                schDttt_re += CustomComplex_real(sch2Dtt) ;
//                schDttt_im += CustomComplex_imag(sch2Dtt) ;
//                schDttt_cor_re += CustomComplex_real(sch2Dtt) ;
//                schDttt_cor_im += CustomComplex_imag(sch2Dtt) ;
//            }
//        }
//    }
//    schDttt_cor = CustomComplex<double> (schDttt_cor_re, schDttt_cor_im);
//
//}
//
//inline void schDttt_corKernel2(CustomComplex<double> &schDttt_cor, int *inv_igp_index, int *indinv, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2)
//{
//    int blkSize = 512;
//    double schDttt_cor_re = 0.00, schDttt_cor_im = 0.00;
//    for(int igbeg = 0; igbeg < ncouls; igbeg += blkSize)
//    {
//        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
//        {
//            for(int ig = igbeg; ig < min(ncouls, igbeg+blkSize); ++ig)
//            {
//                int indigp = inv_igp_index[my_igp] ;
//                int igp = indinv[indigp];
//                CustomComplex<double> sch2Dt = ((I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[ifreq*ncouls*ngpown + my_igp*ncouls + ig]) * fact1 + \
//                                            (I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig]) * fact2) * -0.5;
//                CustomComplex<double> sch2Dtt = aqsntemp[n1*ncouls + ig] * CustomComplex_conj(aqsmtemp[n1*ncouls + igp]) * sch2Dt * vcoul[igp];
//                schDttt_cor_re += CustomComplex_real(sch2Dtt) ;
//                schDttt_cor_im += CustomComplex_imag(sch2Dtt) ;
//            }
//        }
//    }
//    schDttt_cor = CustomComplex<double> (schDttt_cor_re, schDttt_cor_im);
//}

int main(int argc, char** argv)
{

    if(argc != 7)
    {
        cout << "Incorrect Parameters!!! The correct form is " << endl;
        cout << "./a.out number_bands nvband ncouls ngpown nFreq nfreqeval " << endl;
        exit(0);
    }

    int number_bands = atoi(argv[1]);
    int nvband = atoi(argv[2]);
    int ncouls = atoi(argv[3]);
    int ngpown = atoi(argv[4]);
    int nFreq = atoi(argv[5]);
    int nfreqeval = atoi(argv[6]);

    if(ngpown > ncouls)
    {
        cout << "Incorrect Parameters!!! ngpown cannot be greater than ncouls. The correct form is " << endl;
        cout << "./a.out number_bands nvband ncouls ngpown nFreq nfreqeval " << endl;
        exit(0);
    }

    timeval startTimer_Kernel, endTimer_Kernel, \
        start_achsDtemp_Kernel, end_achsDtemp_Kernel, \
        start_asxDtemp_Kernel, end_asxDtemp_Kernel, \
        start_achDtemp_Kernel, end_achDtemp_Kernel,
        start_achDtemp_cor_Kernel, end_achDtemp_cor_Kernel, \
        start_preKernel, end_preKernel;

    gettimeofday(&start_preKernel, NULL);

#if CUDA_VER
    cout << "Cuda Version" << endl;
#else
    cout << "Seq Version" << endl;
#endif

    cout << "number_bands = " << number_bands << \
        "\n nvband = " << nvband << \
        "\n ncouls = " << ncouls << \
        "\n ngpown = " << ngpown << \
        "\n nFreq = " << nFreq << \
        "\n nfreqeval = " << nfreqeval << endl;

    CustomComplex<double> expr0( 0.0 , 0.0);
    CustomComplex<double> expr( 0.5 , 0.5);
    CustomComplex<double> expR( 0.5 , 0.5);
    CustomComplex<double> expA( 0.5 , -0.5);
    CustomComplex<double> exprP1( 0.5 , 0.1);
    double pref_zb = 0.5 / 3.14;

//Start to allocate the data structures;
    int *inv_igp_index = new int[ngpown];
    int *indinv = new int[ncouls];
    double *vcoul = new double[ncouls];
    double *ekq = new double[number_bands];
    double *dFreqGrid = new double[nFreq];
    double *pref = new double[nFreq];
    long double mem_alloc = 0.00;

    CustomComplex<double> *aqsntemp = new CustomComplex<double>[number_bands * ncouls];
    mem_alloc += (number_bands * ncouls * sizeof(CustomComplex<double>));

    CustomComplex<double> *aqsmtemp= new CustomComplex<double>[number_bands * ncouls];
    mem_alloc += (number_bands * ncouls * sizeof(CustomComplex<double>));

    CustomComplex<double> *I_epsR_array = new CustomComplex<double>[nFreq * ngpown * ncouls];
    mem_alloc += (nFreq * ngpown * ncouls * sizeof(CustomComplex<double>));

    CustomComplex<double> *I_epsA_array = new CustomComplex<double>[nFreq * ngpown * ncouls];
    mem_alloc += (nFreq * ngpown * ncouls * sizeof(CustomComplex<double>));

    CustomComplex<double> *schDi = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *sch2Di = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *schDi_cor = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *schDi_corb = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *achDtemp = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *ach2Dtemp = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *achDtemp_cor = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *achDtemp_corb = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *asxDtemp = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *dFreqBrd = new CustomComplex<double>[nFreq];
    mem_alloc += (nfreqeval * 9 * sizeof(CustomComplex<double>));
    mem_alloc += (nFreq * sizeof(CustomComplex<double>)) ;

    CustomComplex<double> *schDt_matrix = new CustomComplex<double>[number_bands * nFreq];
    mem_alloc += (nFreq * number_bands * sizeof(CustomComplex<double>));


    //Variables used for Cuda:
    double *asxDtemp_re = new double[nfreqeval];
    double *asxDtemp_im = new double[nfreqeval];
    double *achDtemp_cor_re = new double[nfreqeval];
    double *achDtemp_cor_im = new double[nfreqeval];

#if CUDA_VER
    double *achsDtemp_re = new double;
    double *achsDtemp_im = new double;
    //Allocated Memory on Device
    CustomComplex<double> *d_aqsmtemp, *d_aqsntemp, \
        *d_I_epsR_array, *d_I_epsA_array, *d_schDt_matrix, \
        *d_achsDtemp, *d_ach2Dtemp , *d_achDtemp_corb;

    int *d_inv_igp_index, *d_indinv;
    double *d_vcoul, *d_dFreqGrid, *d_achsDtemp_re, *d_achsDtemp_im, \
        *d_ekq, *d_asxDtemp_re, *d_asxDtemp_im, *d_achDtemp_cor_re, *d_achDtemp_cor_im;

    CudaSafeCall(cudaMallocManaged((void**) &d_aqsmtemp, number_bands*ncouls*sizeof(CustomComplex<double>)));
    CudaSafeCall(cudaMallocManaged((void**) &d_aqsntemp, number_bands*ncouls*sizeof(CustomComplex<double>)));
    CudaSafeCall(cudaMallocManaged((void**) &d_I_epsR_array, nFreq*ngpown*ncouls*sizeof(CustomComplex<double>)));
    CudaSafeCall(cudaMallocManaged((void**) &d_I_epsA_array, nFreq*ngpown*ncouls*sizeof(CustomComplex<double>)));
    CudaSafeCall(cudaMallocManaged((void**) &d_schDt_matrix, number_bands*nFreq*sizeof(CustomComplex<double>)));
    CudaSafeCall(cudaMallocManaged((void**) &d_inv_igp_index, ngpown*sizeof(int)));
    CudaSafeCall(cudaMallocManaged((void**) &d_indinv, ncouls*sizeof(int)));
    CudaSafeCall(cudaMallocManaged((void**) &d_vcoul, ncouls*sizeof(double)));
    CudaSafeCall(cudaMallocManaged((void**) &d_dFreqGrid, nFreq*sizeof(double)));
    CudaSafeCall(cudaMallocManaged((void**) &d_ekq, number_bands*sizeof(double)));
    CudaSafeCall(cudaMallocManaged((void**) &d_ach2Dtemp, nFreq*sizeof(CustomComplex<double>)));
    CudaSafeCall(cudaMallocManaged((void**) &d_achDtemp_cor_re, nfreqeval*sizeof(double)));
    CudaSafeCall(cudaMallocManaged((void**) &d_achDtemp_cor_im, nfreqeval*sizeof(double)));
    CudaSafeCall(cudaMallocManaged((void**) &d_achDtemp_corb, nfreqeval*sizeof(CustomComplex<double>)));

    //Output
    CudaSafeCall(cudaMallocManaged((void**) &d_achsDtemp, sizeof(CustomComplex<double>)));
    CudaSafeCall(cudaMallocManaged((void**) &d_achsDtemp_re, sizeof(double)));
    CudaSafeCall(cudaMallocManaged((void**) &d_achsDtemp_im, sizeof(double)));
    CudaSafeCall(cudaMallocManaged((void**) &d_asxDtemp_re, nfreqeval*sizeof(double)));
    CudaSafeCall(cudaMallocManaged((void**) &d_asxDtemp_im, nfreqeval*sizeof(double)));

#endif

    const double freqevalmin = 0.00;
    const double freqevalstep = 0.50;
    double dw = -10;
    double occ = 1.00;

    //Initialize the data structures
    for(int ig = 0; ig < ngpown; ++ig)
        inv_igp_index[ig] = ig;

    for(int ig = 0; ig < ncouls; ++ig)
        indinv[ig] = ig;

    for(int i=0; i<number_bands; ++i)
    {
        ekq[i] = dw;
        dw += 1.00;

        for(int j=0; j<ncouls; ++j)
        {
            aqsmtemp[i*ncouls+j] = expr;
            aqsntemp[i*ncouls+j] = expr;
        }

        for(int j=0; j<nFreq; ++j)
            schDt_matrix[i*nFreq + j] = expr0;
    }

    for(int i=0; i<ncouls; ++i)
        vcoul[i] = 1.00;

    for(int i=0; i<nFreq; ++i)
    {
        for(int j=0; j<ngpown; ++j)
        {
            for(int k=0; k<ncouls; ++k)
            {
                I_epsR_array[i*ngpown*ncouls + j * ncouls + k] = expR;
                I_epsA_array[i*ngpown*ncouls + j * ncouls + k] = expA;
            }
        }
    }

    dw = 0.00;
    for(int ijk = 0; ijk < nFreq; ++ijk)
    {
        dFreqBrd[ijk] = exprP1;
        dFreqGrid[ijk] = dw;
        dw += 2.00;
    }

    for(int ifreq = 0; ifreq < nFreq; ++ifreq)
    {
        if(ifreq < nFreq-1)
            pref[ifreq] = (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]) / 3.14;
            else
                pref[ifreq] = pref[ifreq-1];

    }
    pref[0] *= 0.5; pref[nFreq-1] *= 0.5;

    for(int i = 0; i < nfreqeval; ++i)
    {
        schDi[i] = expr0;
        sch2Di[i] = expr0;
        schDi_corb[i] = expr0;
        schDi_cor[i] = expr0;
        asxDtemp[i] = expr0;
        achDtemp[i] = expr0;
        ach2Dtemp[i] = expr0;
        achDtemp_cor[i] = expr0;
        achDtemp_corb[i] = expr0;
        achDtemp_cor_re[i] = 0.00;
        achDtemp_cor_im[i] = 0.00;
    }
    CustomComplex<double> *ssxDittt = new CustomComplex<double>;

#if CUDA_VER
    //Copy data-strudtures value from host to device
    CudaSafeCall(cudaMemcpy(d_aqsmtemp, aqsmtemp, number_bands*ncouls*sizeof(CustomComplex<double>), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_aqsntemp, aqsntemp, number_bands*ncouls*sizeof(CustomComplex<double>), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_I_epsR_array, I_epsR_array, nFreq*ngpown*ncouls*sizeof(CustomComplex<double>), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_I_epsA_array, I_epsA_array, nFreq*ngpown*ncouls*sizeof(CustomComplex<double>), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_schDt_matrix, schDt_matrix, number_bands*nFreq*sizeof(CustomComplex<double>), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_indinv, indinv, ncouls*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_inv_igp_index, inv_igp_index, ngpown*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_vcoul, vcoul, ncouls*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_dFreqGrid, dFreqGrid, nFreq*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_ekq, ekq, number_bands*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_achDtemp_corb, achDtemp_corb, nfreqeval*sizeof(CustomComplex<double>), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_achDtemp_cor_re, achDtemp_cor_re, nfreqeval*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_achDtemp_cor_im, achDtemp_cor_im, nfreqeval*sizeof(double), cudaMemcpyHostToDevice));
#endif

    gettimeofday(&end_preKernel, NULL);
    double elapsed_preKernel = elapsedTime(start_preKernel, end_preKernel);
    cout << "pre kernel time taken = " << elapsed_preKernel << " secs" << endl;

    cout << "starting Kernels" << endl;
    gettimeofday(&startTimer_Kernel, NULL);

    /***********achsDtemp Kernel ****************/
    gettimeofday(&start_achsDtemp_Kernel, NULL);
#if !CUDA_VER
    CustomComplex<double> achsDtemp(0.00, 0.00);
    achsDtemp_Kernel(number_bands, ngpown, ncouls, inv_igp_index, indinv, aqsntemp, aqsmtemp, I_epsR_array, vcoul, achsDtemp);
#else
    d_achsDtemp_Kernel(number_bands, ngpown, ncouls, d_inv_igp_index, d_indinv, d_aqsntemp, d_aqsmtemp, d_I_epsR_array, d_vcoul, d_achsDtemp_re, d_achsDtemp_im);
    cudaDeviceSynchronize();
    CudaSafeCall(cudaMemcpy(achsDtemp_re, d_achsDtemp_re, sizeof(double), cudaMemcpyDeviceToHost));
#endif
    gettimeofday(&end_achsDtemp_Kernel, NULL);

    /***********asxDtemp Kernel ****************/
    gettimeofday(&start_asxDtemp_Kernel, NULL);
#if !CUDA_VER
    asxDtemp_Kernel(nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, occ, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, asxDtemp);
#else
//    d_asxDtemp_Kernel(nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, occ, d_ekq, d_dFreqGrid, d_inv_igp_index, d_indinv, d_aqsmtemp, d_aqsntemp, d_vcoul, d_I_epsR_array, d_I_epsA_array, d_asxDtemp_re, d_asxDtemp_im);
    cudaDeviceSynchronize();
#endif
    gettimeofday(&end_asxDtemp_Kernel, NULL);

    /***********achDtemp Kernel ****************/
//   achDtemp_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, ekq, pref_zb, pref, dFreqGrid, dFreqBrd, schDt_matrix, schDi, schDi_cor, sch2Di, asxDtemp);

    /***********achDtemp_cor Kernel ****************/
    gettimeofday(&start_achDtemp_cor_Kernel, NULL);
#if !CUDA_VER
    achDtemp_cor_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, schDi_cor, schDi_corb, sch2Di, ach2Dtemp, achDtemp_cor, achDtemp_corb);
#else
//    d_achDtemp_cor_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, d_ekq, d_dFreqGrid, d_inv_igp_index, d_indinv, d_aqsmtemp, d_aqsntemp, d_vcoul, d_I_epsR_array, d_I_epsA_array, d_ach2Dtemp, d_achDtemp_cor_re, d_achDtemp_cor_im, d_achDtemp_corb);
    cudaDeviceSynchronize();
#endif
    gettimeofday(&end_achDtemp_cor_Kernel, NULL);

#if CUDA_VER
//    cudaDeviceSynchronize();
    CudaSafeCall(cudaMemcpy(achsDtemp_re, d_achsDtemp_re, sizeof(double), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(achsDtemp_im, d_achsDtemp_im, sizeof(double), cudaMemcpyDeviceToHost));
    CustomComplex<double> achsDtemp(*achsDtemp_re, *achsDtemp_im);

    CudaSafeCall(cudaMemcpy(asxDtemp_re, d_asxDtemp_re, nfreqeval*sizeof(double), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(asxDtemp_im, d_asxDtemp_im, nfreqeval*sizeof(double), cudaMemcpyDeviceToHost));

    CudaSafeCall(cudaMemcpy(achDtemp_cor_re, d_achDtemp_cor_re, nfreqeval*sizeof(double), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(achDtemp_cor_im, d_achDtemp_cor_im, nfreqeval*sizeof(double), cudaMemcpyDeviceToHost));

    for(int iw = 0; iw < nfreqeval; ++iw)
    {
        asxDtemp[iw] = CustomComplex<double>(asxDtemp_re[iw], asxDtemp_im[iw]);
        achDtemp_cor[iw] = CustomComplex<double>(achDtemp_cor_re[iw], achDtemp_cor_im[iw]);
    }
#endif

    gettimeofday(&endTimer_Kernel, NULL);
    double elapsed_achsDtemp = elapsedTime(start_achsDtemp_Kernel, end_achsDtemp_Kernel);
    double elapsed_asxDtemp = elapsedTime(start_asxDtemp_Kernel, end_asxDtemp_Kernel);
    double elapsed_achDtemp_cor = elapsedTime(start_achDtemp_cor_Kernel, end_achDtemp_cor_Kernel);
    double elapsedTimer_Kernel = elapsedTime(startTimer_Kernel, endTimer_Kernel);

    cout << "achsDtemp = " ;
    achsDtemp.print();
    cout << "asxDtemp = " ;
    asxDtemp[0].print();
    cout << "achDtemp_cor = " ;
    achDtemp_cor[0].print();

    cout << "********** achsDtemp Time Taken **********= " << elapsed_achsDtemp << " secs" << endl;
    cout << "********** asxDtemp Time Taken **********= " << elapsed_asxDtemp << " secs" << endl;
    cout << "********** achDtemp_cor Time Taken **********= " << elapsed_achDtemp_cor << " secs" << endl;
    cout << "********** Kernel Time Taken **********= " << elapsedTimer_Kernel << " secs" << endl;

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
    cudaFree(d_achsDtemp_re);
    cudaFree(d_achsDtemp_im);
    cudaFree(d_asxDtemp_re);
    cudaFree(d_asxDtemp_im);
    cudaFree(d_achDtemp_cor_re);
    cudaFree(d_achDtemp_cor_im);
#endif

//Free the allocated memory
    free(aqsntemp);
    free(aqsmtemp);
    free(I_epsA_array);
    free(I_epsR_array);
    free(inv_igp_index);
    free(indinv);
    free(vcoul);
    free(ekq);
    free(dFreqGrid);
    free(pref);
    free(schDi);
    free(sch2Di);
    free(schDi_cor);
    free(schDi_corb);
    free(achDtemp);
    free(ach2Dtemp);
    free(achDtemp_cor);
    free(achDtemp_corb);
    free(asxDtemp);
    free(asxDtemp_re);
    free(asxDtemp_im);
    free(dFreqBrd);
    free(schDt_matrix);
    free(achDtemp_cor_re);
    free(achDtemp_cor_im);

    return 0;
}
