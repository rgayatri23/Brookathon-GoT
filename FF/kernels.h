#include <thrust/complex.h>
#include <iostream>
#include <chrono>

typedef double REAL;
using GPUComplex = thrust::complex<REAL>;

//GPP Function definition
void noflagOCC_cudaKernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, REAL *wx_array, GPUComplex *wtilde_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *I_eps_array, REAL *vcoul, REAL *achtemp_re, REAL *achtemp_im);

//FF Function definition
inline void schDttt_corKernel1(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex &schDttt, REAL *vcoul, int ncouls, int ifreq, int ngpown, int n1, REAL fact1, REAL fact2);

inline void schDttt_corKernel2(GPUComplex &schDttt_cor, int *inv_igp_index, int *indinv, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, REAL *vcoul, int ncouls, int ifreq, int ngpown, int n1, REAL fact1, REAL fact2);

void d_achsDtemp_Kernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, GPUComplex *aqsntemp, GPUComplex *aqsmtemp, GPUComplex *I_epsR_array, REAL *vcoul, REAL *achsDtemp_re, REAL *achsDtemp_im);

void d_asxDtemp_Kernel(int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, REAL freqevalmin, REAL freqevalstep, REAL occ, REAL *ekq, REAL *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, REAL *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, REAL *asxDtemp_re, REAL *asxDtemp_im);

void d_achDtemp_cor_Kernel(int number_bands, int nvband, int nfreqeval, int ncouls, int ngpown, int nFreq, REAL freqevalmin, REAL freqevalstep, REAL *ekq, REAL *dFreqGrid, int *inv_igp_index, int *indinv, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, REAL *vcoul, GPUComplex *I_epsR_array, GPUComplex *I_epsA_array, GPUComplex *ach2Dtemp, REAL *achDtemp_cor_re, REAL *achDtemp_cor_im, GPUComplex *achDtemp_corb);