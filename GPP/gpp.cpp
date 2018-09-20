#include "../ComplexClass/CustomComplex.h"
#include <openacc.h>

using namespace std;
#define nstart 0
#define nend 3

#define reductionVersion 1

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

inline void reduce_achstemp(int number_bands, int* inv_igp_index, int ncouls, CustomComplex<double>  *aqsmtemp, CustomComplex<double> *aqsntemp, CustomComplex<double> *I_eps_array,\
        CustomComplex<double> achstemp,  int* indinv, int ngpown, double* vcoul, int numThreads)
{
    double to1 = 1e-6;
    CustomComplex<double> schstemp(0.0, 0.0);;

#pragma acc kernels present(inv_igp_index, aqsmtemp, aqsntemp, I_eps_array, indinv, vcoul)
    for(int n1 = 0; n1<number_bands; ++n1)
        for(int my_igp = 0; my_igp< ngpown; my_igp++)
        {
            CustomComplex<double> schs(0.0, 0.0);
            CustomComplex<double> matngmatmgp(0.0, 0.0);
            CustomComplex<double> matngpmatmg(0.0, 0.0);
            CustomComplex<double> mygpvar1(0.00, 0.00), mygpvar2(0.00, 0.00);
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            if(indigp == ncouls)
                igp = ncouls-1;

            if(!(igp > ncouls || igp < 0)){

                mygpvar1 = CustomComplex_conj(aqsmtemp[n1*ncouls+igp]);
                mygpvar2 = aqsntemp[n1*ncouls+igp];
                schs = I_eps_array[my_igp*ncouls+igp];
                matngmatmgp = mygpvar1 * aqsntemp[n1*ncouls+igp];

                if(CustomComplex_abs(schs) > to1)
                    schstemp += matngmatmgp * schs;
                }
                else
                {
                    for(int ig=1; ig<ncouls; ++ig)
                    {
                        CustomComplex<double> mult_result(I_eps_array[my_igp*ncouls+ig] * mygpvar1);
                        schstemp -= aqsntemp[n1*ncouls +igp] * mult_result;
                    }
                }

            schstemp = schstemp * vcoul[igp]*0.5;
            achstemp += schstemp;
        }
}

inline void flagOCC_solver(double wxt, CustomComplex<double> *wtilde_array, int my_igp, int n1, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, CustomComplex<double> *I_eps_array, CustomComplex<double> &ssxt, CustomComplex<double> &scht,int ncouls, int igp, int number_bands, int ngpown)
{
    CustomComplex<double> expr0(0.00, 0.00);
    CustomComplex<double> expr(0.5, 0.5);
    CustomComplex<double> matngmatmgp(0.0, 0.0);
    CustomComplex<double> matngpmatmg(0.0, 0.0);

    for(int ig=0; ig<ncouls; ++ig)
    {
        CustomComplex<double> wtilde = wtilde_array[my_igp*ncouls+ig];
        CustomComplex<double> wtilde2 = wtilde * wtilde;
        CustomComplex<double> Omega2 = wtilde2*I_eps_array[my_igp*ncouls+ig];
        CustomComplex<double> mygpvar1 = CustomComplex_conj(aqsmtemp[n1*ncouls+igp]);
        CustomComplex<double> mygpvar2 = aqsmtemp[n1*ncouls+igp];
        CustomComplex<double> matngmatmgp = aqsntemp[n1*ncouls+ig] * mygpvar1;
        if(ig != igp) matngpmatmg = CustomComplex_conj(aqsmtemp[n1*ncouls+ig]) * mygpvar2;

        double ssxcutoff;
        double to1 = 1e-6;
        double sexcut = 4.0;
        double limitone = 1.0/(to1*4.0);
        double limittwo = pow(0.5,2);
        CustomComplex<double> sch(0.00, 0.00), ssx(0.00, 0.00);

        CustomComplex<double> wdiff = wxt - wtilde;

        CustomComplex<double> cden = wdiff;
        double rden = 1/CustomComplex_real(cden * CustomComplex_conj(cden));
        CustomComplex<double> delw = wtilde * CustomComplex_conj(cden) * rden;
        double delwr = CustomComplex_real(delw * CustomComplex_conj(delw));
        double wdiffr = CustomComplex_real(wdiff * CustomComplex_conj(wdiff));

        if((wdiffr > limittwo) && (delwr < limitone))
        {
            sch = delw * I_eps_array[my_igp*ngpown+ig];
            double cden = std::pow(wxt,2);
            rden = std::pow(cden,2);
            rden = 1.00 / rden;
            ssx = Omega2 * cden * rden;
        }
        else if (delwr > to1)
        {
            sch = expr0;
            cden = wtilde2 * (0.50 + delw) * 4.00;
            rden = CustomComplex_real(cden * CustomComplex_conj(cden));
            rden = 1.00/rden;
            ssx = -Omega2 * CustomComplex_conj(cden) * delw * rden;
        }
        else
        {
            sch = expr0;
            ssx = expr0;
        }

        ssxcutoff = CustomComplex_abs(I_eps_array[my_igp*ngpown+ig]) * sexcut;
        if((CustomComplex_abs(ssx) > ssxcutoff) && (wxt < 0.00)) ssx = expr0;

        ssxt += matngmatmgp * ssx;
        scht += matngmatmgp * sch;
    }
}


void till_nvband(int number_bands, int nvband, int ngpown, int ncouls, CustomComplex<double> *asxtemp, double *wx_array, CustomComplex<double> *wtilde_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, CustomComplex<double> *I_eps_array, int *inv_igp_index, int *indinv, double *vcoul)
{
    const double occ=1.0;
//#pragma omp parallel for collapse(3)
#pragma acc parallel loop present(inv_igp_index, indinv, wtilde_array, wx_array, aqsmtemp, aqsntemp, I_eps_array, vcoul, asxtemp) collapse(3)
    for(int n1 = 0; n1 < nvband; n1++)
    {
         for(int my_igp=0; my_igp<ngpown; ++my_igp)
         {
            for(int iw=nstart; iw<nend; iw++)
            {
                 int indigp = inv_igp_index[my_igp];
                 int igp = indinv[indigp];
                 CustomComplex<double> ssxt(0.00, 0.00);
                 CustomComplex<double> scht(0.00, 0.00);
     //            flagOCC_solver(wx_array[iw], wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, ncouls, igp, number_bands, ngpown);
                 asxtemp[iw] += ssxt * occ * vcoul[igp];
           }
         }
    }
}

void noflagOCC_solver(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, CustomComplex<double> *wtilde_array, CustomComplex<double> *aqsmtemp, \
        CustomComplex<double> *aqsntemp, CustomComplex<double> *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im, CustomComplex<double> *achtemp, double &elapsed_time)
{
    timeval startTimer , endTimer;
    gettimeofday(&startTimer, NULL);

    double achtemp_re0 = 0.00, achtemp_re1 = 0.00, achtemp_re2 = 0.00, \
                         achtemp_im0 = 0.00, achtemp_im1 = 0.00, achtemp_im2 = 0.00;

#pragma acc parallel loop gang present(inv_igp_index, indinv, wtilde_array, wx_array, aqsmtemp, \
     aqsntemp, I_eps_array, vcoul)\
    num_gangs(number_bands) vector_length(32) \
    reduction(+:achtemp_re0, achtemp_re1, achtemp_re2, achtemp_im0, achtemp_im1, achtemp_im2)
    for(int n1 = 0; n1<number_bands; ++n1)
    {
#pragma acc loop vector\
    reduction(+:achtemp_re0, achtemp_re1, achtemp_re2, achtemp_im0, achtemp_im1, achtemp_im2)
        for(int my_igp=0; my_igp<ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];

            CustomComplex<double> wdiff(0.00, 0.00), delw(0.00, 0.00);

            double achtemp_re_loc[nend-nstart], achtemp_im_loc[nend-nstart];
            for(int iw = nstart; iw < nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}

#pragma acc loop seq
            for(int ig = 0; ig<ncouls; ++ig)
            {
#pragma acc loop seq
                for(int iw = nstart; iw < nend; ++iw)
                {
                    wdiff = wx_array[iw] - wtilde_array[my_igp*ncouls+ig];
                    delw = wtilde_array[my_igp*ncouls+ig] * CustomComplex_conj(wdiff) * (1/CustomComplex_real((wdiff * CustomComplex_conj(wdiff))));
                    CustomComplex<double> sch_array = CustomComplex_conj(aqsmtemp[n1*ncouls+igp]) * aqsntemp[n1*ncouls+ig] * delw * I_eps_array[my_igp*ncouls+ig] * 0.5*vcoul[igp];
                    double sch_array_re = CustomComplex_real(sch_array);
                    double sch_array_im = CustomComplex_imag(sch_array);

                    achtemp_re_loc[iw] += sch_array_re ;
                    achtemp_im_loc[iw] += sch_array_im ;
                }
            }

#if reductionVersion
            achtemp_re0 += achtemp_re_loc[0];
            achtemp_re1 += achtemp_re_loc[1];
            achtemp_re2 += achtemp_re_loc[2];
            achtemp_im0 += achtemp_im_loc[0];
            achtemp_im1 += achtemp_im_loc[1];
            achtemp_im2 += achtemp_im_loc[2];
#else
#pragma acc loop seq
            for(int iw = nstart; iw < nend; ++iw)
            {
#pragma acc atomic update
                achtemp_re[iw] += achtemp_re_loc[iw];
#pragma acc atomic update
                achtemp_im[iw] += achtemp_im_loc[iw];
            }
#endif
        } //ngpown
    } //number_bands
#if reductionVersion
    achtemp_re[0] = achtemp_re0;
    achtemp_re[1] = achtemp_re1;
    achtemp_re[2] = achtemp_re2;
    achtemp_im[0] = achtemp_im0;
    achtemp_im[1] = achtemp_im1;
    achtemp_im[2] = achtemp_im2;
#else
#pragma acc update self(achtemp_re[nstart:nend], achtemp_im[nstart:nend])
#endif
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
    CustomComplex<double> achstemp;

    //Print Memory Foot print
    cout << "Memory Foot Print = " << memFootPrint / pow(1024,3) << " GBs" << endl;


    //Initailize structures
    init_structs(number_bands, ngpown, ncouls, aqsmtemp, aqsntemp, I_eps_array, wtilde_array, vcoul, inv_igp_index, indinv, asxtemp, achtemp_re, achtemp_im, wx_array);

    //Start the timer before the work begins.
    timeval startTimer, endTimer;
    gettimeofday(&startTimer, NULL);
    double elapsed_noFlagOCC = 0.00;

    //0-nvband iterations
    till_nvband(number_bands, nvband, ngpown, ncouls, asxtemp, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, inv_igp_index, indinv, vcoul);

    //reduction on achstemp
    reduce_achstemp(number_bands, inv_igp_index, ncouls,aqsmtemp, aqsntemp, I_eps_array, achstemp, indinv, ngpown, vcoul, numThreads);

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
