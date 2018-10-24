#include "../ComplexClass/CustomComplex.h"
#include <openacc.h>
using namespace std;
#define nstart 0
#define nend 3

double elapsedTime(timeval start_time, timeval end_time)
{
    return ((end_time.tv_sec - start_time.tv_sec) +1e-6*(end_time.tv_usec - start_time.tv_usec));
}

void init_structs(size_t number_bands, size_t ngpown, size_t ncouls, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, CustomComplex<double> *I_eps_array, \
        CustomComplex<double> *wtilde_array, double *vcoul, int *inv_igp_index, int *indinv, CustomComplex<double> *asxtemp, double *achtemp_re, double *achtemp_im, double *wx_array)
{
    const double dw = 1;
    const double e_lk = 10;
    const double to1 = 1e-6;
    const double limittwo = pow(0.5,2);
    const double e_n1kq= 6.0;
    CustomComplex<double> expr0(0.00, 0.00);
    CustomComplex<double> expr(0.5, 0.5);

#pragma acc enter data create (aqsmtemp[0:number_bands*ncouls], aqsntemp[0:number_bands*ncouls], I_eps_array[0:ngpown*ncouls], vcoul[0:ncouls], inv_igp_index[0:ngpown], \
        indinv[0:ncouls+1], asxtemp[nstart:nend], achtemp_re[nstart:nend], achtemp_im[nstart:nend], wtilde_array[0:ngpown*ncouls], wx_array[nstart:nend])

#pragma acc parallel loop copyin(expr) present(aqsmtemp, aqsntemp)
   for(int i=0; i<number_bands; i++)
       for(int j=0; j<ncouls; j++)
       {
           aqsmtemp[i*ncouls+j] = expr;
           aqsntemp[i*ncouls+j] = expr;
       }

#pragma acc parallel loop copyin(expr) present(I_eps_array, wtilde_array)
   for(int i=0; i<ngpown; i++)
       for(int j=0; j<ncouls; j++)
       {
           I_eps_array[i*ncouls+j] = expr;
           wtilde_array[i*ncouls+j] = expr;
       }

#pragma acc parallel loop present(vcoul)
   for(int i=0; i<ncouls; i++)
       vcoul[i] = 1.0;


#pragma acc parallel loop present(inv_igp_index)
    for(int ig=0; ig < ngpown; ++ig)
        inv_igp_index[ig] = (ig+1) * ncouls / ngpown;

#pragma acc parallel loop present(indinv)
    for(int ig=0; ig<ncouls; ++ig)
        indinv[ig] = ig;
        indinv[ncouls] = ncouls-1;

#pragma acc parallel loop present(asxtemp, achtemp_re, achtemp_im)
       for(int iw=nstart; iw<nend; ++iw)
       {
           asxtemp[iw] = expr0;
           achtemp_re[iw] = 0.00;
           achtemp_im[iw] = 0.00;
           wx_array[iw] = e_lk - e_n1kq + dw*((iw+1)-2);
           if(wx_array[iw] < to1) wx_array[iw] = to1;
       }
}

void noflagOCC_solver(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, CustomComplex<double> *wtilde_array, CustomComplex<double> *aqsmtemp, \
        CustomComplex<double> *aqsntemp, CustomComplex<double> *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im, CustomComplex<double> *achtemp, double &elapsed_time)
{
    timeval startTimer , endTimer;
    gettimeofday(&startTimer, NULL);

    double achtemp_re0 = 0.00, achtemp_re1 = 0.00, achtemp_re2 = 0.00, \
                         achtemp_im0 = 0.00, achtemp_im1 = 0.00, achtemp_im2 = 0.00;

#pragma acc parallel loop gang collapse(2) present(inv_igp_index, indinv, wtilde_array, wx_array, aqsmtemp, \
     aqsntemp, I_eps_array, vcoul)\
    reduction(+:achtemp_re0, achtemp_re1, achtemp_re2, achtemp_im0, achtemp_im1, achtemp_im2)//\
    num_gangs(number_bands*ngpown) vector_length(32)
    for(int my_igp=0; my_igp<ngpown; ++my_igp)
    {
        for(int n1 = 0; n1<number_bands; ++n1)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];

#pragma acc loop vector\
    reduction(+:achtemp_re0, achtemp_re1, achtemp_re2, achtemp_im0, achtemp_im1, achtemp_im2)
            for(int ig = 0; ig<ncouls; ++ig)
            {
                double achtemp_re_loc[nend-nstart], achtemp_im_loc[nend-nstart];
                for(int iw = nstart; iw < nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}
#pragma acc loop seq
                for(int iw = nstart; iw < nend; ++iw)
                {
                    CustomComplex<double> wdiff = wx_array[iw] - wtilde_array[my_igp*ncouls+ig];
                    CustomComplex<double> delw = wtilde_array[my_igp*ncouls+ig] * CustomComplex_conj(wdiff) * (1/CustomComplex_real((wdiff * CustomComplex_conj(wdiff))));
                    CustomComplex<double> sch_array = CustomComplex_conj(aqsmtemp[n1*ncouls+igp]) * aqsntemp[n1*ncouls+ig] * delw * I_eps_array[my_igp*ncouls+ig] * 0.5*vcoul[igp];
                    double sch_array_re = CustomComplex_real(sch_array);
                    double sch_array_im = CustomComplex_imag(sch_array);

                    achtemp_re_loc[iw] = sch_array_re ;
                    achtemp_im_loc[iw] = sch_array_im ;

                }
                achtemp_re0 += achtemp_re_loc[0];
                achtemp_re1 += achtemp_re_loc[1];
                achtemp_re2 += achtemp_re_loc[2];
                achtemp_im0 += achtemp_im_loc[0];
                achtemp_im1 += achtemp_im_loc[1];
                achtemp_im2 += achtemp_im_loc[2];
            }

        } //ngpown
    } //number_bands

    //Store back reduction variables
    achtemp_re[0] = achtemp_re0;
    achtemp_re[1] = achtemp_re1;
    achtemp_re[2] = achtemp_re2;
    achtemp_im[0] = achtemp_im0;
    achtemp_im[1] = achtemp_im1;
    achtemp_im[2] = achtemp_im2;
    gettimeofday(&endTimer, NULL);
    elapsed_time = elapsedTime(startTimer, endTimer);

    for(int iw = nstart; iw < nend; ++iw)
        achtemp[iw] = CustomComplex<double>(achtemp_re[iw], achtemp_im[iw]);

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
    const double limittwo = pow(0.5,2);

    //OpenMP Printing of threads on Host and Device
    int tid, numThreads;
#pragma omp parallel shared(numThreads) private(tid)
    {
        tid = omp_get_thread_num();
        if(tid == 0)
            numThreads = omp_get_num_threads();
    }
    std::cout << "Number of OpenMP Threads = " << numThreads << endl;

    //Printing out the params passed.
    std::cout << "Sizeof(CustomComplex<double> = " << sizeof(CustomComplex<double>) << " bytes" << std::endl;
    std::cout << "number_bands = " << number_bands \
        << "\t nvband = " << nvband \
        << "\t ncouls = " << ncouls \
        << "\t nodes_per_group  = " << nodes_per_group \
        << "\t ngpown = " << ngpown \
        << "\t nend = " << nend \
        << "\t nstart = " << nstart << endl;

    CustomComplex<double> expr0(0.00, 0.00);
    CustomComplex<double> expr(0.5, 0.5);
    long double memFootPrint = 0.00;

    //ALLOCATE statements from fortran gppkernel.
    CustomComplex<double> *achtemp = new CustomComplex<double>[nend-nstart];
    CustomComplex<double> *asxtemp = new CustomComplex<double>[nend-nstart];
    memFootPrint += 2*(nend-nstart)*sizeof(CustomComplex<double>);

    CustomComplex<double> *aqsmtemp = new CustomComplex<double>[number_bands*ncouls];
    CustomComplex<double> *aqsntemp = new CustomComplex<double>[number_bands*ncouls];
    memFootPrint += 2*(number_bands*ncouls)*sizeof(CustomComplex<double>);

    CustomComplex<double> *I_eps_array = new CustomComplex<double>[ngpown*ncouls];
    CustomComplex<double> *wtilde_array = new CustomComplex<double>[ngpown*ncouls];
    memFootPrint += 2*(ngpown*ncouls)*sizeof(CustomComplex<double>);

    double *vcoul = new double[ncouls];
    memFootPrint += ncouls*sizeof(double);

    int *inv_igp_index = new int[ngpown];
    int *indinv = new int[ncouls+1];
    memFootPrint += ngpown*sizeof(int);
    memFootPrint += (ncouls+1)*sizeof(int);

//Real and imaginary parts of achtemp calculated separately to avoid critical.
    double *achtemp_re = new double[nend-nstart];
    double *achtemp_im = new double[nend-nstart];
    memFootPrint += 2*(nend-nstart)*sizeof(double);
    double wx_array[nend-nstart];
    //Print Memory Foot print
    cout << "Memory Foot Print = " << memFootPrint / pow(1024,3) << " GBs" << endl;

    //Initailize structures
    init_structs(number_bands, ngpown, ncouls, aqsmtemp, aqsntemp, I_eps_array, wtilde_array, vcoul, inv_igp_index, indinv, asxtemp, achtemp_re, achtemp_im, wx_array);

    //Start the timer before the work begins.
    timeval startTimer, endTimer;
    gettimeofday(&startTimer, NULL);
    double elapsed_noFlagOCC = 0.00;
    //main-loop with output on achtemp divide among achtemp_re && achtemp_im
    noflagOCC_solver(number_bands, ngpown, ncouls, inv_igp_index, indinv, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, vcoul, achtemp_re, achtemp_im, achtemp, \
            elapsed_noFlagOCC);

    //Time Taken
    gettimeofday(&endTimer, NULL);
    double elapsedTimer = (endTimer.tv_sec - startTimer.tv_sec) +1e-6*(endTimer.tv_usec - startTimer.tv_usec);

    printf("\n Final achtemp\n");
    achtemp[0].print();

    cout << "********** noFlagOCC Time Taken **********= " << elapsed_noFlagOCC << " secs" << endl;
    cout << "********** Kernel Time Taken **********= " << elapsedTimer << " secs" << endl;

    free(achtemp);
    free(aqsmtemp);
    free(aqsntemp);
    free(I_eps_array);
    free(wtilde_array);
    free(asxtemp);
    free(vcoul);
    free(inv_igp_index);
    free(indinv);

    return 0;
}
