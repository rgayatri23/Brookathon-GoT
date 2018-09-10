#include "../ComplexClass/GPUComplex.h"
#include "../ComplexClass/GPUComplex.h"
#include "../ComplexClass/cudaAlloc.h"

#define CUDA_VER 1

using namespace std;
#define nstart 0
#define nend 3

void noflagOCC_solver(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, GPUComplex *wtilde_array, GPUComplex *aqsmtemp, GPUComplex *aqsntemp, GPUComplex *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im)
{
//#pragma omp parallel for  default(shared) firstprivate(ngpown, ncouls, number_bands) reduction(+:achtemp_re[nstart:nend], achtemp_im[nstart:nend])
    for(int n1 = 0; n1<number_bands; ++n1) 
    {
        for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];

            GPUComplex wdiff(0.00, 0.00), delw(0.00, 0.00);

            double achtemp_re_loc[nend-nstart], achtemp_im_loc[nend-nstart];
            for(int iw = nstart; iw < nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}

            for(int ig = 0; ig<ncouls; ++ig)
            {
                for(int iw = nstart; iw < nend; ++iw)
                {
                    wdiff = wx_array[iw] - wtilde_array[my_igp*ncouls+ig]; //2 flops

                    //2 conj + 2 * product + 1 mult + 1 divide = 17
                    delw = wtilde_array[my_igp*ncouls+ig] * GPUComplex_conj(wdiff) * (1/GPUComplex_real((wdiff * GPUComplex_conj(wdiff)))); 

                    //1 conj + 3 product + 1 mult + 1 real-mult = 22
                    GPUComplex sch_array = GPUComplex_conj(aqsmtemp[n1*ncouls+igp]) * aqsntemp[n1*ncouls+ig] * delw * I_eps_array[my_igp*ncouls+ig] * 0.5*vcoul[igp];

                    //2 flops
                    achtemp_re_loc[iw] += GPUComplex_real(sch_array);
                    achtemp_im_loc[iw] += GPUComplex_imag(sch_array);
                }
            }
            for(int iw = nstart; iw < nend; ++iw)
            {
                achtemp_re[iw] += achtemp_re_loc[iw];
                achtemp_im[iw] += achtemp_im_loc[iw];
            }
        } //ngpown
    } //number_bands
}

int main(int argc, char** argv)
{

    if (argc != 5)
    {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <nodes_per_mpi_group> " << endl;
        exit (0);
    }

//Input parameters stored in these variables.
    const int number_bands = atoi(argv[1]);
    const int nvband = atoi(argv[2]);
    const int ncouls = atoi(argv[3]);
    const int nodes_per_group = atoi(argv[4]);
    const int npes = 1; 
    const int ngpown = ncouls / (nodes_per_group * npes); 

//Constants that will be used later
    const double e_lk = 10;
    const double dw = 1;
    const double to1 = 1e-6;
    const double gamma = 0.5;
    const double sexcut = 4.0;
    const double limitone = 1.0/(to1*4.0);
    const double limittwo = pow(0.5,2);
    const double e_n1kq= 6.0; 
    const double occ=1.0;

    //Printing out the params passed.
    std::cout << "Sizeof(GPUComplex = " << sizeof(GPUComplex) << " bytes" << std::endl;
    std::cout << "number_bands = " << number_bands \
        << "\t nvband = " << nvband \
        << "\t ncouls = " << ncouls \
        << "\t nodes_per_group  = " << nodes_per_group \
        << "\t ngpown = " << ngpown \
        << "\t nend = " << nend \
        << "\t nstart = " << nstart << endl;
   
    GPUComplex expr0(0.00, 0.00);
    GPUComplex expr(0.5, 0.5);
    long double memFootPrint = 0.00;

    //ALLOCATE statements from fortran gppkernel.
    GPUComplex *acht_n1_loc = new GPUComplex[number_bands];
    memFootPrint += number_bands*sizeof(GPUComplex);

    GPUComplex *achtemp = new GPUComplex[nend-nstart];
    GPUComplex *asxtemp = new GPUComplex[nend-nstart];
    GPUComplex *ssx_array = new GPUComplex[nend-nstart];
    memFootPrint += 3*(nend-nstart)*sizeof(GPUComplex);

    GPUComplex *aqsmtemp = new GPUComplex[number_bands*ncouls];
    GPUComplex *aqsntemp = new GPUComplex[number_bands*ncouls];
    memFootPrint += 2*(number_bands*ncouls)*sizeof(GPUComplex);

    GPUComplex *I_eps_array = new GPUComplex[ngpown*ncouls];
    GPUComplex *wtilde_array = new GPUComplex[ngpown*ncouls];
    memFootPrint += 2*(ngpown*ncouls)*sizeof(GPUComplex);

    GPUComplex *ssxa = new GPUComplex[ncouls];
    double *vcoul = new double[ncouls];
    memFootPrint += ncouls*sizeof(GPUComplex);
    memFootPrint += ncouls*sizeof(double);

    int *inv_igp_index = new int[ngpown];
    int *indinv = new int[ncouls+1];
    memFootPrint += ngpown*sizeof(int);
    memFootPrint += (ncouls+1)*sizeof(int);

//Real and imaginary parts of achtemp calculated separately to avoid critical.
    double *achtemp_re = new double[nend-nstart];
    double *achtemp_im = new double[nend-nstart];
    memFootPrint += 2*(nend-nstart)*sizeof(double);

#if CUDA_VER
    GPUComplex *d_aqsmtemp, *d_aqsntemp, *d_I_eps_array, *d_wtilde_array;
    int *d_inv_igp_index, *d_indinv; 
    double *d_achtemp_re, *d_achtemp_im, *d_wx_array, *d_vcoul;
    
    CudaSafeCall(cudaMalloc((void**) &d_inv_igp_index, ngpown*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &d_indinv, (ncouls+1)*sizeof(int)));
    CudaSafeCall(cudaMalloc((void**) &d_wx_array, (nend-nstart)*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_wtilde_array, ngpown*ncouls*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_aqsntemp, number_bands*ncouls*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_aqsmtemp, number_bands*ncouls*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_I_eps_array, ngpown*ncouls*sizeof(GPUComplex)));
    CudaSafeCall(cudaMalloc((void**) &d_vcoul, ncouls*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_achtemp_re, (nend-nstart)*sizeof(double)));
    CudaSafeCall(cudaMalloc((void**) &d_achtemp_im, (nend-nstart)*sizeof(double)));
#endif

    double wx_array[nend-nstart];
    GPUComplex achstemp;
                        
    //Print Memory Foot print 
    cout << "Memory Foot Print = " << memFootPrint / pow(1024,3) << " GBs" << endl;


   for(int i=0; i<number_bands; i++)
       for(int j=0; j<ncouls; j++)
       {
           aqsmtemp[i*ncouls+j] = expr;
           aqsntemp[i*ncouls+j] = expr;
       }

   for(int i=0; i<ngpown; i++)
       for(int j=0; j<ncouls; j++)
       {
           I_eps_array[i*ncouls+j] = expr;
           wtilde_array[i*ncouls+j] = expr;
       }

   for(int i=0; i<ncouls; i++)
       vcoul[i] = 1.0;


    for(int ig=0; ig < ngpown; ++ig)
        inv_igp_index[ig] = (ig+1) * ncouls / ngpown;

    //Do not know yet what this array represents
    for(int ig=0; ig<ncouls; ++ig)
        indinv[ig] = ig;
        indinv[ncouls] = ncouls-1;

       for(int iw=nstart; iw<nend; ++iw)
       {
           asxtemp[iw] = expr0;
           achtemp_re[iw] = 0.00;
           achtemp_im[iw] = 0.00;
       }

        for(int iw=nstart; iw<nend; ++iw)
        {
            wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
            if(wx_array[iw] < to1) wx_array[iw] = to1;
        }

    //Start the timer before the work begins.
    timeval startTimer, endTimer;
    gettimeofday(&startTimer, NULL);

#if CUDA_VER
    CudaSafeCall(cudaMemcpy(d_inv_igp_index, inv_igp_index, ngpown*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_indinv, indinv, (ncouls+1)*sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_wx_array, wx_array, (nend-nstart)*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_wtilde_array, wtilde_array, ngpown*ncouls*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_aqsmtemp, aqsmtemp, number_bands*ncouls*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_aqsntemp, aqsntemp, number_bands*ncouls*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_I_eps_array, I_eps_array, ngpown*ncouls*sizeof(GPUComplex), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_vcoul, vcoul, ncouls*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_achtemp_re, achtemp_re, (nend-nstart)*sizeof(double), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_achtemp_im, achtemp_im, (nend-nstart)*sizeof(double), cudaMemcpyHostToDevice));
#endif

#if CUDA_VER
    noflagOCC_cudaKernel(number_bands, ngpown, ncouls, d_inv_igp_index, d_indinv, d_wx_array, d_wtilde_array, d_aqsmtemp, d_aqsntemp, d_I_eps_array, d_vcoul, d_achtemp_re, d_achtemp_im);
    cudaDeviceSynchronize();
    CudaSafeCall(cudaMemcpy(achtemp_re, d_achtemp_re, (nend-nstart)*sizeof(double), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(achtemp_im, d_achtemp_im, (nend-nstart)*sizeof(double), cudaMemcpyDeviceToHost));

#else
    noflagOCC_solver(number_bands, ngpown, ncouls, inv_igp_index, indinv, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, vcoul, achtemp_re, achtemp_im);
#endif

    //Time Taken
    gettimeofday(&endTimer, NULL);
    double elapsedTimer = (endTimer.tv_sec - startTimer.tv_sec) +1e-6*(endTimer.tv_usec - startTimer.tv_usec);

    printf(" \n Final achstemp\n");
    achstemp.print();

    printf("\n Final achtemp\n");

    for(int iw=nstart; iw<nend; ++iw)
    {
        GPUComplex tmp(achtemp_re[iw], achtemp_im[iw]);
        achtemp[iw] = tmp;
//        achtemp[iw].print();
    }
        achtemp[0].print();

    cout << "********** Kernel Time Taken **********= " << elapsedTimer << " secs" << endl;

#if CUDA_VER
    cudaFree(d_inv_igp_index);
    cudaFree(d_indinv);
    cudaFree(d_wx_array);
    cudaFree(d_wtilde_array);
    cudaFree(d_aqsmtemp);
    cudaFree(d_aqsntemp);
    cudaFree(d_I_eps_array);
    cudaFree(d_vcoul);
    cudaFree(d_achtemp_re);
    cudaFree(d_achtemp_im);


#endif

    free(acht_n1_loc);
    free(achtemp);
    free(aqsmtemp);
    free(aqsntemp);
    free(I_eps_array);
    free(wtilde_array);
    free(asxtemp);
    free(vcoul);
    free(ssx_array);
    free(inv_igp_index);
    free(indinv);

    return 0;
}
