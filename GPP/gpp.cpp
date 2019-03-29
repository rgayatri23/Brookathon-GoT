#include "../ComplexClass/CustomComplex.h"

#define nstart 0
#define nend 3

inline void short_correntess(CustomComplex<double> result)
{
    double re_diff = result.get_real() - -63015.384615;
    double im_diff = result.get_imag() - 315076.923077;

    if(re_diff < 0.00001 && im_diff < 0.00001)
        printf("\n!!!! SUCCESS - !!!! Test Correctness test passed :-D :-D\n\n");
    else
        printf("\n!!!! FAILURE - Correctness test failed :-( :-(  \n");
}

inline void correntess(CustomComplex<double> result)
{
    double re_diff = result.get_real() - -264241151.997370;
    double im_diff = result.get_imag() - 1321205760.015211;

    if(re_diff < 0.00001 && im_diff < 0.00001)
        printf("\n!!!! SUCCESS - !!!! Correctness test passed :-D :-D\n\n");
    else
        printf("\n!!!! FAILURE - Correctness test failed :-( :-(  \n");
}

inline void reduce_achstemp(int n1, int number_bands, int* inv_igp_index, int ncouls, CustomComplex<double>  *aqsmtemp, CustomComplex<double> *aqsntemp, CustomComplex<double> *I_eps_array, CustomComplex<double> achstemp,  int* indinv, int ngpown, double* vcoul, int numThreads)
{
    double to1 = 1e-6;
    CustomComplex<double> schstemp(0.0, 0.0);;

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
#pragma omp parallel for collapse(3)
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
                 flagOCC_solver(wx_array[iw], wtilde_array, my_igp, n1, aqsmtemp, aqsntemp, I_eps_array, ssxt, scht, ncouls, igp, number_bands, ngpown);
                 asxtemp[iw] += ssxt * occ * vcoul[igp];
           }
         }
    }
}

void noflagOCC_solver(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, double *wx_array, CustomComplex<double> *wtilde_array, CustomComplex<double> *aqsmtemp, CustomComplex<double> *aqsntemp, CustomComplex<double> *I_eps_array, double *vcoul, double *achtemp_re, double *achtemp_im, double &elapsedKernelTimer)
{
    timeval startKernelTimer, endKernelTimer;
    //Vars to use for reduction
    double ach_re0 = 0.00, ach_re1 = 0.00, ach_re2 = 0.00, \
        ach_im0 = 0.00, ach_im1 = 0.00, ach_im2 = 0.00;
    gettimeofday(&startKernelTimer, NULL);

#if defined(_OPENMP)
#pragma omp parallel for\
    reduction(+:ach_re0, ach_re1, ach_re2, ach_im0, ach_im1, ach_im2)
#endif
    for(int my_igp=0; my_igp<ngpown; ++my_igp)
    {
        for(int n1 = 0; n1<number_bands; ++n1)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            CustomComplex<double> sch_store1 = CustomComplex_conj(aqsmtemp[n1*ncouls+igp])*  aqsntemp[n1*ncouls+igp] * 0.5*vcoul[igp];

            double achtemp_re_loc[nend-nstart], achtemp_im_loc[nend-nstart];
            for(int iw = nstart; iw < nend; ++iw) {achtemp_re_loc[iw] = 0.00; achtemp_im_loc[iw] = 0.00;}
//            CustomComplex<double> delw(0.0, 0.0), wdiff(0.0, 0.0), sch_array(0.0, 0.0);


            [&](){

            for(int ig = 0; ig<ncouls; ++ig)
            {
                for(int iw = nstart; iw < nend; ++iw)
                {
                    CustomComplex<double> wdiff = wx_array[iw] - wtilde_array[my_igp*ncouls+ig];
                    CustomComplex<double> delw = wtilde_array[my_igp*ncouls + ig]* CustomComplex_conj(wdiff) * (1/CustomComplex_real((wdiff * CustomComplex_conj(wdiff))));
                    CustomComplex<double> sch_array = delw  * I_eps_array[my_igp*ncouls +ig] * sch_store1;

                    achtemp_re_loc[iw] += CustomComplex_real(sch_array);
                    achtemp_im_loc[iw] += CustomComplex_imag(sch_array);
                }
            }
            };
            ach_re0 += achtemp_re_loc[0];
            ach_re1 += achtemp_re_loc[1];
            ach_re2 += achtemp_re_loc[2];
            ach_im0 += achtemp_im_loc[0];
            ach_im1 += achtemp_im_loc[1];
            ach_im2 += achtemp_im_loc[2];
        } //ngpown
    } //number_bands

    gettimeofday(&endKernelTimer, NULL);
    elapsedKernelTimer = (endKernelTimer.tv_sec - startKernelTimer.tv_sec) +1e-6*(endKernelTimer.tv_usec - startKernelTimer.tv_usec);

    achtemp_re[0] = ach_re0;
    achtemp_re[1] = ach_re1;
    achtemp_re[2] = ach_re2;
    achtemp_im[0] = ach_im0;
    achtemp_im[1] = ach_im1;
    achtemp_im[2] = ach_im2;
}

int main(int argc, char** argv)
{

    if (argc != 5)
    {
        std::cout << "The correct form of input is : " << endl;
        std::cout << " ./a.out <number_bands> <number_valence_bands> <number_plane_waves> <nodes_per_mpi_group> " << endl;
        exit (0);
    }

#if defined(_OPENMP)
    cout << "\n ************OpenMP 3.0**********\n" << endl;
#endif

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

    //Start the timer before the work begins.
    double elapsedKernelTimer, elapsedTimer;
    timeval startTimer, endTimer;
    gettimeofday(&startTimer, NULL);


    //OpenMP Printing of threads on Host and Device
    int tid, numThreads, numTeams;

#if defined(_OPENMP)
#pragma omp parallel shared(numThreads) private(tid)
    {
        tid = omp_get_thread_num();
        if(tid == 0)
            numThreads = omp_get_num_threads();
    }
    std::cout << "Number of OpenMP Threads = " << numThreads << endl;
#endif

#if defined(OPENMP_TARGET)
#pragma omp target map(tofrom: numTeams, numThreads)
#pragma omp teams shared(numTeams) private(tid)
    {
        tid = omp_get_team_num();
        if(tid == 0)
        {
            numTeams = omp_get_num_teams();
#pragma omp parallel
            {
                int ttid = omp_get_thread_num();
                if(ttid == 0)
                    numThreads = omp_get_num_threads();
            }
        }
    }
    std::cout << "Number of OpenMP Teams = " << numTeams << std::endl;
    std::cout << "Number of OpenMP DEVICE Threads = " << numThreads << std::endl;
#endif

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
    CustomComplex<double> *acht_n1_loc = new CustomComplex<double>[number_bands];
    memFootPrint += number_bands*sizeof(CustomComplex<double>);

    CustomComplex<double> *achtemp = new CustomComplex<double>[nend-nstart];
    CustomComplex<double> *asxtemp = new CustomComplex<double>[nend-nstart];
    CustomComplex<double> *ssx_array = new CustomComplex<double>[nend-nstart];
    memFootPrint += 3*(nend-nstart)*sizeof(CustomComplex<double>);

    CustomComplex<double> *aqsmtemp = new CustomComplex<double>[number_bands*ncouls];
    CustomComplex<double> *aqsntemp = new CustomComplex<double>[number_bands*ncouls];
    memFootPrint += 2*(number_bands*ncouls)*sizeof(CustomComplex<double>);

    CustomComplex<double> *I_eps_array = new CustomComplex<double>[ngpown*ncouls];
    CustomComplex<double> *wtilde_array = new CustomComplex<double>[ngpown*ncouls];
    memFootPrint += 2*(ngpown*ncouls)*sizeof(CustomComplex<double>);

    CustomComplex<double> *ssxa = new CustomComplex<double>[ncouls];
    double *vcoul = new double[ncouls];
    memFootPrint += ncouls*sizeof(CustomComplex<double>);
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

    //Initailize OpenACC structures
    //0-nvband iterations
    till_nvband(number_bands, nvband, ngpown, ncouls, asxtemp, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, inv_igp_index, indinv, vcoul);

    //reduction on achstemp
#pragma omp parallel for
    for(int n1 = 0; n1<number_bands; ++n1)
        reduce_achstemp(n1, number_bands, inv_igp_index, ncouls,aqsmtemp, aqsntemp, I_eps_array, achstemp, indinv, ngpown, vcoul, numThreads);

    noflagOCC_solver(number_bands, ngpown, ncouls, inv_igp_index, indinv, wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array, vcoul, achtemp_re, achtemp_im, elapsedKernelTimer);

    gettimeofday(&endTimer, NULL);
    elapsedTimer = (endTimer.tv_sec - startTimer.tv_sec) +1e-6*(endTimer.tv_usec - startTimer.tv_usec);

    for(int iw=nstart; iw<nend; ++iw)
    {
        CustomComplex<double> tmp(achtemp_re[iw], achtemp_im[iw]);
        achtemp[iw] = tmp;
    }


    //Check for correctness
    if(ncouls == 32768)
        correntess(achtemp[0]);
    else
        short_correntess(achtemp[0]);
//    printf("\n Final achtemp\n");
        achtemp[0].print();

    cout << "********** Kernel Time Taken **********= " << elapsedKernelTimer << " secs" << endl;
    cout << "********** Total Time Taken **********= " << elapsedTimer << " secs" << endl;

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
