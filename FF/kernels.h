#include <thrust/complex.h>
#include <iostream>
#include <chrono>

using GPUComplex = thrust::complex<double>;

//GPP Function definition
void noflagOCC_cudaKernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, GPUComplex *wtilde_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im);

//FF Function definition
inline void schDttt_corKernel1(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex &schDttt, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2);

inline void schDttt_corKernel2(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, int ncouls, int ifreq, int ngpown, int n1, double fact1, double fact2);

void d_achsDtemp_Kernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, double *vcoul, double *achsDtemp_re, double *achsDtemp_im);

void d_asxDtemp_Kernel(int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double occ, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, double *asxDtemp_re, double *asxDtemp_im);

void d_achDtemp_cor_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, double freqevalmin, double freqevalstep, double *ekq, double *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, double *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *ach2Dtemp, double *achDtemp_cor_re, double *achDtemp_cor_im, GPUComplex *achDtemp_corb);