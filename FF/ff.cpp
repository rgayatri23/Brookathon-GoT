#include "../ComplexClass/CustomComplex.h"
#include <accel.h>

double elapsedTime(timeval start_time, timeval end_time)
{
    return ((end_time.tv_sec - start_time.tv_sec) +1e-6*(end_time.tv_usec - start_time.tv_usec));
}

static inline void schDttt_corKernel1(CustomComplex<double> &schDttt_cor, int *inv_igp_index, int *indinv, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2);

static inline void schDttt_corKernel2(CustomComplex<double> &schDttt_cor, int *inv_igp_index, int *indinv, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2);

void init_structs(size_t nfreqeval, size_t nFreq, size_t ngpown, size_t ncouls, size_t number_bands, int* inv_igp_index, int *indinv, double *ekq, double *pref, CustomComplex<double>* aqsmtemp, CustomComplex<double>* aqsntemp, \
            CustomComplex<double>* schDt_matrix, double *vcoul,CustomComplex<double>* I_epsR_array,CustomComplex<double>* I_epsA_array,CustomComplex<double>* dFreqBrd, double *dFreqGrid, \
            CustomComplex<double>* asxDtemp,CustomComplex<double>* achDtemp,CustomComplex<double>* achDtemp_cor, double* achDtemp_cor_re, double* achDtemp_cor_im, \
            double *asxDtemp_re, double *asxDtemp_im)
{
    double dw = -10;
    CustomComplex<double> expr0( 0.0 , 0.0);
    CustomComplex<double> expr( 0.5 , 0.5);
    CustomComplex<double> expR( 0.5 , 0.5);
    CustomComplex<double> expA( 0.5 , -0.5);
    CustomComplex<double> exprP1( 0.5 , 0.1);
#pragma acc enter data create(ekq[0:number_bands], inv_igp_index[0:ngpown], \
        I_epsR_array[0:nFreq*ngpown*ncouls], I_epsA_array[0:nFreq*ngpown*ncouls], \
        indinv[0:ncouls], aqsmtemp[0:number_bands*ncouls], aqsntemp[0:number_bands*ncouls], \
	schDt_matrix[0:number_bands*nFreq], vcoul[0:ncouls], dFreqBrd[0:nFreq], \
	asxDtemp[0:nfreqeval], achDtemp[0:nfreqeval], \
        achDtemp_cor[0:nfreqeval], achDtemp_cor_re[0:nfreqeval], achDtemp_cor_im[0:nfreqeval], dFreqGrid[0:nFreq],pref[0:nFreq])

#pragma acc parallel loop
    for(int ig = 0; ig < ngpown; ++ig)
        inv_igp_index[ig] = ig;

#pragma acc parallel loop
    for(int ig = 0; ig < ncouls; ++ig)
        indinv[ig] = ig;

#pragma acc parallel loop
    for(int i=0; i<number_bands; ++i)
        ekq[i] = dw+i;

#pragma acc parallel loop copyin(expr) present(aqsmtemp,aqsntemp)
    for(int i=0; i<number_bands; ++i)
    {
        for(int j=0; j<ncouls; ++j)
        {
            aqsmtemp[i*ncouls+j] = expr;
            aqsntemp[i*ncouls+j] = expr;
        }
    }

#pragma acc parallel loop  present(schDt_matrix)
    for(int i=0; i<number_bands; ++i)
    {
        for(int j=0; j<nFreq; ++j)
            schDt_matrix[i*nFreq + j] = expr0;
    }

#pragma acc parallel loop
    for(int i=0; i<ncouls; ++i)
        vcoul[i] = 1.00;

#pragma acc parallel loop present(I_epsR_array,I_epsA_array)
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
#pragma acc parallel loop present(dFreqBrd[0:nFreq],dFreqGrid[0:nFreq])
    for(int ijk = 0; ijk < nFreq; ++ijk)
    {
        dFreqBrd[ijk] = exprP1;
        dFreqGrid[ijk] = (ijk*2.0);
    }

#pragma acc parallel loop present(dFreqGrid[0:nFreq],pref[0:nFreq])
    for(int ifreq = 0; ifreq < nFreq; ++ifreq)
    {
        if(ifreq < nFreq-1)
            pref[ifreq] = (dFreqGrid[ifreq+1] - dFreqGrid[ifreq]) / 3.14;
            else
                pref[ifreq] = pref[ifreq-1];

    }
#pragma acc serial
    {
    pref[0] *= 0.5; pref[nFreq-1] *= 0.5;
    }

#pragma acc parallel loop present(asxDtemp,achDtemp,achDtemp_cor,achDtemp_cor_re,achDtemp_cor_im, asxDtemp_re, asxDtemp_im)
    for(int i = 0; i < nfreqeval; ++i)
    {
        asxDtemp[i] = expr0;
        achDtemp[i] = expr0;
        achDtemp_cor[i] = expr0;
        achDtemp_cor_re[i] = 0.00;
        achDtemp_cor_im[i] = 0.00;
        asxDtemp_re[i] = 0.00;
        asxDtemp_im[i] = 0.00;
    }

}

void exit_data(size_t nfreqeval, size_t nFreq, size_t ngpown, size_t ncouls, size_t number_bands, int* inv_igp_index, int *indinv, double *ekq, double *pref, CustomComplex<double>* aqsmtemp, CustomComplex<double>* aqsntemp, \
            CustomComplex<double>* schDt_matrix, double *vcoul,CustomComplex<double>* I_epsR_array,CustomComplex<double>* I_epsA_array,CustomComplex<double>* dFreqBrd, double *dFreqGrid, \
            CustomComplex<double>* asxDtemp,CustomComplex<double>* achDtemp,CustomComplex<double>* achDtemp_cor, double* achDtemp_cor_re, double* achDtemp_cor_im, \
            double *asxDtemp_re, double *asxDtemp_im)
{
#pragma acc exit data delete(ekq[0:number_bands], inv_igp_index[0:ngpown], \
        I_epsR_array[0:nFreq*ngpown*ncouls], I_epsA_array[0:nFreq*ngpown*ncouls], \
        indinv[0:ncouls], aqsmtemp[0:number_bands*ncouls], aqsntemp[0:number_bands*ncouls], \
	schDt_matrix[0:number_bands*nFreq], vcoul[0:ncouls], dFreqBrd[0:nFreq], \
	asxDtemp[0:nfreqeval], achDtemp[0:nfreqeval], \
        achDtemp_cor[0:nfreqeval], achDtemp_cor_re[0:nfreqeval], achDtemp_cor_im[0:nfreqeval], dFreqGrid[0:nFreq],pref[0:nFreq], \
        asxDtemp_re[0:nfreqeval], asxDtemp_im[0:nfreqeval])
}
void calculate_schDt_lin3(CustomComplex<double>& schDt_lin3, bool flag_occ, int freqevalmin, double *ekq, int iw, int freqevalstep, double cedifft_zb_right, double cedifft_zb_left, CustomComplex<double> schDt_left, CustomComplex<double> schDt_lin2, int n1, double pref_zb, CustomComplex<double> pref_zb_compl, CustomComplex<double> schDt_avg)
{
    double intfact = (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_right) / (freqevalmin - ekq[n1] + (iw-1) * freqevalstep - cedifft_zb_left);
    if(intfact < 0.0001) intfact = 0.0001;
    if(intfact > 10000) intfact = 10000;
    intfact = -log(intfact);
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

static inline void compute_fact(double wx, size_t nFreq, double *dFreqGrid, double &fact1, double &fact2, int &ifreq, int loop, bool flag_occ)
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

static inline void ssxDittt_kernel(int *inv_igp_index, int *indinv, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, CustomComplex<double> *I_eps_array, CustomComplex<double> &ssxDittt, int ngpown, int ncouls, int n1,int ifreq, double fact1, double fact2, int igp, int my_igp)
{
    double ssxDitt_re = 0.00, ssxDitt_im = 0.00;
    int blkSize = 512;

    //Constant variables used in every iteration: should technically be done by the compiler
    const int counter1 = ifreq*ngpown*ncouls;
    const int counter2 = (ifreq+1)*ngpown*ncouls;
    const int counter3 = my_igp*ncouls;
    const int counter4 = n1*ncouls;
#pragma acc loop vector \
    reduction(+:ssxDitt_re, ssxDitt_im)
    for(int igbeg = 0; igbeg < ncouls; igbeg += blkSize)
    {
        for(int ig = igbeg; ig < min(igbeg+blkSize,ncouls); ++ig) //ncouls = 33401
        {
            CustomComplex<double> ssxDit = I_eps_array[counter1 + counter3 + ig] * fact1 + \
                                         I_eps_array[counter2 + counter3 + ig] * fact2;
            CustomComplex<double> ssxDitt = aqsntemp[counter4 + ig] * CustomComplex_conj(aqsmtemp[counter4 + igp]) * ssxDit * vcoul[igp];
            ssxDitt_re += CustomComplex_real(ssxDitt);
            ssxDitt_im += CustomComplex_imag(ssxDitt);
        }
    }
    ssxDittt = CustomComplex<double> (ssxDitt_re, ssxDitt_im);
}

void achsDtemp_Kernel(int number_bands, int ngpown, int ncouls, int nFreq, int *inv_igp_index, int *indinv, CustomComplex<double> *aqsntemp, CustomComplex<double> *aqsmtemp, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array,double *vcoul, CustomComplex<double> &achsDtemp, double &elapsed_time)
{
    double achsDtemp_re = 0.00, achsDtemp_im = 0.00;
    timeval startTimer , endTimer;

    gettimeofday(&startTimer, NULL);
#pragma acc parallel loop gang collapse(2) present(inv_igp_index, indinv, aqsmtemp, aqsntemp, I_epsR_array, I_epsA_array, vcoul)\
    reduction(+:achsDtemp_re, achsDtemp_im) \
    num_gangs(number_bands) num_workers(1) vector_length(128) async(1)
    for(int n1 = 0; n1 < number_bands; ++n1) //number_bands = 15023
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp) //ngpown = 66
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            double schsDtemp_re = 0.00, schsDtemp_im = 0.00;
            CustomComplex<double> aqsmtemp_conj = CustomComplex_conj(aqsmtemp[n1*ncouls + igp]);
            const int counter1 = 1*ngpown*ncouls + my_igp*ncouls;
            const int counter2 = n1*ncouls;

#pragma acc loop vector \
            reduction(-:schsDtemp_re, schsDtemp_im)
            for(int ig = 0; ig < ncouls; ++ig) //ncouls = 33401
            {
                schsDtemp_re -= CustomComplex_real(aqsntemp[counter2 + ig] * aqsmtemp_conj * I_epsR_array[counter1 + ig]);
                schsDtemp_im -= CustomComplex_imag(aqsntemp[counter2 + ig] * aqsmtemp_conj * I_epsR_array[counter1 + ig]);
            }

            achsDtemp_re += schsDtemp_re * vcoul[igp] * 0.5;
            achsDtemp_im += schsDtemp_im * vcoul[igp] * 0.5;
        }
    } //n1
    achsDtemp = CustomComplex<double> (achsDtemp_re, achsDtemp_im) ;

    gettimeofday(&endTimer, NULL);
    elapsed_time = elapsedTime(startTimer, endTimer);
}

void asxDtemp_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double occ, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, double *asxDtemp_re, double *asxDtemp_im, double &elapsed_time)
{
    timeval startTimer , endTimer;
    gettimeofday(&startTimer, NULL);
#pragma acc parallel loop gang vector collapse(3)  \
     present(ekq, asxDtemp_re, asxDtemp_im, inv_igp_index, indinv, aqsmtemp, aqsntemp, \
             I_epsR_array, I_epsA_array,asxDtemp_re,asxDtemp_im,dFreqGrid[:nFreq],vcoul) async(2)
    for(int n1 = 0; n1 < nvband; ++n1) //nvband=1998
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp) //ngpown = 66
        {
            for(int iw = 0; iw < nfreqeval; ++iw) //nfreqeval = 10
            {
                double wx = freqevalmin - ekq[n1] + freqevalstep;
                int indigp = inv_igp_index[my_igp];
                int igp = indinv[indigp];
                CustomComplex<double> ssxDittt(0.00, 0.00);
                int ifreq = 0;
                double fact1 = 0.00, fact2 = 0.00;

                compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 1, 0);

            if(wx > 0)
                ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2, igp, my_igp);
                else
                    ssxDittt_kernel(inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsA_array, ssxDittt, ngpown, ncouls, n1, ifreq, fact1, fact2, igp, my_igp);

                double ssxDittt_re = CustomComplex_real(ssxDittt * occ);
                double ssxDittt_im = CustomComplex_imag(ssxDittt * occ);
#pragma acc atomic
                asxDtemp_re[iw] += ssxDittt_re;
#pragma acc atomic
                asxDtemp_im[iw] += ssxDittt_im;
            } // iw
        }
    }

#pragma acc update self(asxDtemp_re[0:nfreqeval],asxDtemp_im[0:nfreqeval])
    gettimeofday(&endTimer, NULL);
    elapsed_time = elapsedTime(startTimer, endTimer);

}


void achDtemp_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double pref_zb, double *pref, double *dFreqGrid, CustomComplex<double> *dFreqBrd, CustomComplex<double> *schDt_matrix, CustomComplex<double> *achDtemp)
{
    bool flag_occ;
    CustomComplex<double> expr0(0.00, 0.00);
    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        for(int ifreq = 0; ifreq < nFreq; ++ifreq)
        {
            flag_occ = n1 < nvband;
            CustomComplex<double> schDt = schDt_matrix[n1*nFreq + ifreq];
            double cedifft_zb = dFreqGrid[ifreq];
            double cedifft_zb_right, cedifft_zb_left;
            CustomComplex<double> schDt_right, schDt_left, schDt_avg, schDt_lin, schDt_lin2, schDt_lin3;
            CustomComplex<double> cedifft_compl(cedifft_zb, 0.00);
            CustomComplex<double> cedifft_cor;
            CustomComplex<double> cedifft_coh = cedifft_compl - dFreqBrd[ifreq];
            CustomComplex<double> pref_zb_compl(0.00, pref_zb);

            if(flag_occ)
                cedifft_cor = cedifft_compl * -1 - dFreqBrd[ifreq];
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
                    calculate_schDt_lin3(schDt_lin3, flag_occ, freqevalmin, ekq, iw, freqevalstep, cedifft_zb_right, cedifft_zb_left, schDt_left, schDt_lin2, n1, pref_zb, pref_zb_compl, schDt_avg);

                    schDt_lin3 += schDt_lin;
                }
            }

            for(int iw = 0; iw < nfreqeval; ++iw)
            {
                CustomComplex<double> schDi_elem(0.00, 0.00);
                double wx = freqevalmin - ekq[n1] + (iw-1) * freqevalstep;
                CustomComplex<double> tmp(0.00, pref[ifreq]);
                schDi_elem = schDi_elem - ((tmp*schDt) / (wx- cedifft_coh));
                achDtemp[iw] += schDi_elem;
            }
        }
    }

}

void achDtemp_cor_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, double *achDtemp_cor_re, double *achDtemp_cor_im, double &elapsed_time)
{
    bool flag_occ;
    timeval startTimer , endTimer;
    gettimeofday(&startTimer, NULL);

#pragma acc parallel loop gang vector collapse(2) present(inv_igp_index[0:ngpown], indinv[0:ncouls], aqsmtemp[0:number_bands*ncouls], aqsntemp[0:number_bands*ncouls], I_epsR_array[0:nFreq*ngpown*ncouls], I_epsA_array[0:nFreq*ngpown*ncouls],dFreqGrid,vcoul)\
    copyout(achDtemp_cor_re[0:nfreqeval], achDtemp_cor_im[0:nfreqeval]) \
    num_gangs(number_bands) num_workers(1) vector_length(32) async(3)
    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        for(int iw = 0; iw < nfreqeval; ++iw)
        {
            flag_occ = n1 < nvband;
            CustomComplex<double> schDi_cor(0.00, 0.00);
            double wx = freqevalmin - ekq[n1] + freqevalstep;

            double fact1 = 0.00, fact2 = 0.00;
            int ifreq = 0.00;

            compute_fact(wx, nFreq, dFreqGrid, fact1, fact2, ifreq, 2, flag_occ);

            if(wx > 0)
            {
                if(!flag_occ)
                    schDttt_corKernel1(schDi_cor, inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2);
            }
            else if(flag_occ)
                schDttt_corKernel2(schDi_cor, inv_igp_index, indinv, I_epsR_array, I_epsA_array, aqsmtemp, aqsntemp, vcoul,  ncouls, ifreq, ngpown, n1, fact1, fact2);


//Summing up at the end of iw loop
            double schDi_cor_re = CustomComplex_real(schDi_cor);
            double schDi_cor_im = CustomComplex_imag(schDi_cor);
#pragma acc atomic
            achDtemp_cor_re[iw] += schDi_cor_re;
#pragma acc atomic
            achDtemp_cor_im[iw] += schDi_cor_im;
        }// iw
    } //n1

    gettimeofday(&endTimer, NULL);
    elapsed_time = elapsedTime(startTimer, endTimer);
#pragma acc update self(achDtemp_cor_re[0:nfreqeval],achDtemp_cor_im[0:nfreqeval])
}

static inline void schDttt_corKernel1(CustomComplex<double> &schDttt_cor, int *inv_igp_index, int *indinv, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2)
{
    int blkSize = 512;
    double schDttt_cor_re = 0.00, schDttt_cor_im = 0.00, \
        schDttt_re = 0.00, schDttt_im = 0.00;
    for(int igbeg = 0; igbeg < ncouls; igbeg += blkSize)
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            for(int ig = igbeg; ig < min(ncouls, igbeg+blkSize); ++ig)
            {
                int indigp = inv_igp_index[my_igp] ;
                int igp = indinv[indigp];
                CustomComplex<double> sch2Dt = (I_epsR_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[ifreq*ngpown*ncouls + my_igp*ncouls + ig]) * fact1 + \
                                            (I_epsR_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig] - I_epsA_array[(ifreq+1)*ngpown*ncouls + my_igp*ncouls + ig]) * fact2;
                CustomComplex<double> sch2Dtt = aqsntemp[n1*ncouls + ig] * CustomComplex_conj(aqsmtemp[n1*ncouls + igp]) * sch2Dt * vcoul[igp];


                schDttt_re += CustomComplex_real(sch2Dtt) ;
                schDttt_im += CustomComplex_imag(sch2Dtt) ;
                schDttt_cor_re += CustomComplex_real(sch2Dtt) ;
                schDttt_cor_im += CustomComplex_imag(sch2Dtt) ;
            }
        }
    }
    schDttt_cor = CustomComplex<double> (schDttt_cor_re, schDttt_cor_im);
}

static inline void schDttt_corKernel2(CustomComplex<double> &schDttt_cor, int *inv_igp_index, int *indinv, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2)
{
    int blkSize = 512;

    //Constant variables used in every iteration: should technically be done by the compiler
    const int counter1 = ifreq*ngpown*ncouls;
    const int counter2 = (ifreq+1)*ngpown*ncouls;
    const int counter4 = n1*ncouls;
    double schDttt_cor_re = 0.00, schDttt_cor_im = 0.00;
//#pragma acc loop vector \
    reduction(+:schDttt_cor_re, schDttt_cor_im)
    for(int igbeg = 0; igbeg < ncouls; igbeg += blkSize)
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            const int counter3 = my_igp*ncouls;
            for(int ig = igbeg; ig < min(ncouls, igbeg+blkSize); ++ig)
            {
                int indigp = inv_igp_index[my_igp] ;
                int igp = indinv[indigp];
                CustomComplex<double> sch2Dt = ((I_epsR_array[counter1 + counter3 + ig] - I_epsA_array[counter1 + counter3 + ig]) * fact1 + \
                                            (I_epsR_array[counter2 + counter3 + ig] - I_epsA_array[counter2 + counter3 + ig]) * fact2) * -0.5;
                CustomComplex<double> sch2Dtt = aqsntemp[counter4 + ig] * CustomComplex_conj(aqsmtemp[counter4 + igp]) * sch2Dt * vcoul[igp];
                schDttt_cor_re += CustomComplex_real(sch2Dtt) ;
                schDttt_cor_im += CustomComplex_imag(sch2Dtt) ;
            }
        }
    }
    schDttt_cor = CustomComplex<double> (schDttt_cor_re, schDttt_cor_im);
}

void all_Kernels(size_t number_bands, size_t ngpown, size_t ncouls, size_t nFreq, size_t nvband, size_t nfreqeval, int *inv_igp_index, int *indinv, CustomComplex<double> *aqsntemp, \
        CustomComplex<double> *aqsmtemp, CustomComplex<double> *I_epsR_array, CustomComplex<double> *I_epsA_array, double *vcoul, CustomComplex<double> &achsDtemp, \
            CustomComplex<double> *asxDtemp, CustomComplex<double> *achDtemp_cor, double *achDtemp_cor_re, double *achDtemp_cor_im, \
            double freqevalmin, double freqevalstep, double occ, double *ekq, double *dFreqGrid, CustomComplex<double> *dFreqBrd, \
            double *pref, const double pref_zb, CustomComplex<double> *schDt_matrix, double *asxDtemp_re, double *asxDtemp_im, \
            double &elapsed_achsDtemp, double &elapsed_asxDtemp, double &elapsed_achDtemp_cor)
{
    /***********achsDtemp Kernel ****************/
    achsDtemp_Kernel(number_bands, ngpown, ncouls, nFreq, inv_igp_index, indinv, aqsntemp, aqsmtemp, I_epsR_array, I_epsA_array, vcoul, achsDtemp, elapsed_achsDtemp);

    /***********asxDtemp Kernel ****************/
    asxDtemp_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, occ, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp,\
            vcoul, I_epsR_array, I_epsA_array, asxDtemp_re, asxDtemp_im, elapsed_asxDtemp);

    /***********achDtemp Kernel ****************/
    achDtemp_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, ekq, pref_zb, pref, dFreqGrid, dFreqBrd, schDt_matrix, asxDtemp);

    /***********achDtemp_cor Kernel ****************/
    achDtemp_cor_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, achDtemp_cor_re, achDtemp_cor_im, elapsed_achDtemp_cor);

//Aggregate results
#pragma acc kernels async(3) wait(2)
    for(int iw = 0; iw < nfreqeval; ++iw)
    {
        asxDtemp[iw] = CustomComplex<double>(asxDtemp_re[iw], asxDtemp_im[iw]);
        achDtemp_cor[iw] = CustomComplex<double>(achDtemp_cor_re[iw], achDtemp_cor_im[iw]);
    }
#pragma acc wait
}

int main(int argc, char** argv)
{

    if(argc != 7)
    {
        cout << "Incorrect Parameters!!! The correct form is " << endl;
        cout << "./a.out number_bands nvband ncouls ngpown nFreq nfreqeval " << endl;
        exit(0);
    }

    const size_t number_bands = atoi(argv[1]);
    const size_t nvband = atoi(argv[2]);
    const size_t ncouls = atoi(argv[3]);
    const size_t ngpown = atoi(argv[4]);
    const size_t nFreq = atoi(argv[5]);
    const size_t nfreqeval = atoi(argv[6]);

    const double freqevalmin = 0.00;
    const double freqevalstep = 0.50;
    const double occ = 1.00;
    const double pref_zb = 0.5 / 3.14;
    double dw = -10;

    if(ngpown > ncouls)
    {
        cout << "Incorrect Parameters!!! ngpown cannot be greater than ncouls. The correct form is " << endl;
        cout << "./a.out number_bands nvband ncouls ngpown nFreq nfreqeval " << endl;
        exit(0);
    }

    timeval startTimer_Kernel, endTimer_Kernel, \
        start_preKernel, end_preKernel;

    gettimeofday(&start_preKernel, NULL);

    cout << "\n number_bands = " << number_bands << \
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

    CustomComplex<double> *achDtemp = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *achDtemp_cor = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *asxDtemp = new CustomComplex<double>[nfreqeval];
    CustomComplex<double> *dFreqBrd = new CustomComplex<double>[nFreq];
    mem_alloc += (nfreqeval * 3 * sizeof(CustomComplex<double>));
    mem_alloc += (nFreq * sizeof(CustomComplex<double>)) ;

    CustomComplex<double> *schDt_matrix = new CustomComplex<double>[number_bands * nFreq];
    mem_alloc += (nFreq * number_bands * sizeof(CustomComplex<double>));

    //Variables used :
    CustomComplex<double> achsDtemp(0.00, 0.00);

    //OpenACC specific separation of data structures
    double *achDtemp_cor_re = new double[nfreqeval];
    double *achDtemp_cor_im = new double[nfreqeval];
    double *asxDtemp_re = new double[nfreqeval];
    double *asxDtemp_im = new double[nfreqeval];
    mem_alloc += nfreqeval * 4 * sizeof(double);


    init_structs(nfreqeval, nFreq, ngpown, ncouls, number_bands, inv_igp_index, indinv, ekq, pref, aqsmtemp, aqsntemp, \
            schDt_matrix, vcoul, I_epsR_array, I_epsA_array, dFreqBrd, dFreqGrid, \
            asxDtemp, achDtemp, achDtemp_cor, achDtemp_cor_re, achDtemp_cor_im, \
            asxDtemp_re, asxDtemp_im);


    gettimeofday(&end_preKernel, NULL);
    double elapsed_preKernel = elapsedTime(start_preKernel, end_preKernel);

    cout << "Memory Used = " << mem_alloc/(1024 * 1024 * 1024) << " GB" << endl;
    cout << "pre kernel time taken = " << elapsed_preKernel << " secs" << endl;

    gettimeofday(&startTimer_Kernel, NULL);
    cout << "starting Kernels" << endl;

    double elapsed_achsDtemp = 0.00, elapsed_asxDtemp = 0.00, elapsed_achDtemp_cor = 0.00;

    all_Kernels(number_bands, ngpown, ncouls, nFreq, nvband, nfreqeval, inv_igp_index, indinv, aqsntemp, aqsmtemp, I_epsR_array, I_epsA_array, vcoul, achsDtemp, \
            asxDtemp, achDtemp_cor, achDtemp_cor_re, achDtemp_cor_im, \
            freqevalmin, freqevalstep, occ, ekq, dFreqGrid, dFreqBrd,\
            pref, pref_zb, schDt_matrix, asxDtemp_re, asxDtemp_im, \
            elapsed_achsDtemp, elapsed_asxDtemp, elapsed_achDtemp_cor);

//    /***********achsDtemp Kernel ****************/
//    achsDtemp_Kernel(number_bands, ngpown, ncouls, nFreq, inv_igp_index, indinv, aqsntemp, aqsmtemp, I_epsR_array, I_epsA_array, vcoul, achsDtemp, elapsed_achsDtemp);
//
//    /***********asxDtemp Kernel ****************/
//    asxDtemp_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, occ, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, asxDtemp, elapsed_asxDtemp);
//
//    /***********achDtemp Kernel ****************/
//    achDtemp_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, ekq, pref_zb, pref, dFreqGrid, dFreqBrd, schDt_matrix, asxDtemp);
//
//    /***********achDtemp_cor Kernel ****************/
//    achDtemp_cor_Kernel(number_bands, nvband, nfreqeval, ncouls, ngpown, nFreq, freqevalmin, freqevalstep, ekq, dFreqGrid, inv_igp_index, indinv, aqsmtemp, aqsntemp, vcoul, I_epsR_array, I_epsA_array, achDtemp_cor_re, achDtemp_cor_im, elapsed_achDtemp_cor);
//
//    for(int iw = 0; iw < nfreqeval; ++iw)
//        achDtemp_cor[iw] = CustomComplex<double>(achDtemp_cor_re[iw], achDtemp_cor_im[iw]);

    exit_data(nfreqeval, nFreq, ngpown, ncouls, number_bands, inv_igp_index, indinv, ekq, pref, aqsmtemp, aqsntemp, \
            schDt_matrix, vcoul, I_epsR_array, I_epsA_array, dFreqBrd, dFreqGrid, \
            asxDtemp, achDtemp, achDtemp_cor, achDtemp_cor_re, achDtemp_cor_im, asxDtemp_re, asxDtemp_im);

    gettimeofday(&endTimer_Kernel, NULL);
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

//Free the allocated memory
   delete(aqsntemp);
   delete(aqsmtemp);
   delete(I_epsA_array);
   delete(I_epsR_array);
   delete(inv_igp_index);
   delete(indinv);
   delete(vcoul);
   delete(ekq);
   delete(dFreqGrid);
   delete(pref);
   delete(achDtemp);
   delete(achDtemp_cor_re);
   delete(achDtemp_cor_im);
   delete(achDtemp_cor);
   delete(asxDtemp);
   delete(dFreqBrd);
   delete(schDt_matrix);
   delete(asxDtemp_re);
   delete(asxDtemp_im);

    return 0;
}
