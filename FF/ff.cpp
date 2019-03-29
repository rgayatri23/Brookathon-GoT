#include "../ComplexClass/CustomComplex.h"

void achsDtemp_Kernel(int number_bands, int ngpown, int ncouls, int *inv_igp_index, int *indinv, \
        CustomComplex<double> *aqsntemp, CustomComplex<double> *aqsmtemp, double *vcoul, \
        CustomComplex<double> &achsDtemp, CustomComplex<double> *I_epsR_array)
{
    double achsDtemp_re = 0.00, achsDtemp_im = 0.00;
//#pragma omp parallel for default(shared) reduction(+:achsDtemp_re, achsDtemp_im)
    for(int n1 = 0; n1 < number_bands; ++n1)
    {
        for(int my_igp = 0; my_igp < ngpown; ++my_igp)
        {
            int indigp = inv_igp_index[my_igp];
            int igp = indinv[indigp];
            CustomComplex<double> schsDtemp(0.00, 0.00);

            for(int ig = 0; ig < ncouls; ++ig)
                schsDtemp = schsDtemp - \
                            aqsntemp[n1*ncouls + ig] * CustomComplex_conj(aqsmtemp[n1*ncouls + igp]) * \
                            I_epsR_array[ngpown*ncouls + my_igp*ncouls + ig] * vcoul[igp] * 0.5;

//schsDtemp = schsDtemp -
//    aqsntemp[n1*ncouls + ig] * CustomComplex_conj(aqsmtemp[n1*ncouls + igp]) *
//    I_epsR_array[1*ngpown*ncouls + my_igp*ncouls + ig];

            achsDtemp_re += CustomComplex_real(schsDtemp);
            achsDtemp_im += CustomComplex_imag(schsDtemp);
        } //my_igp
    } //n1
    achsDtemp = CustomComplex<double> (achsDtemp_re, achsDtemp_im) ;

}

int main(int argc, char** argv)
{
    int number_bands, nvband, ncouls, ngpown, nFreq, nfreqeval;

    if(argc == 1)
    {
        number_bands = 15023;
        nvband = 199;
        ncouls = 23401;
        ngpown = 66;
        nFreq = 15;
        nfreqeval = 10;
    }
    else if(argc == 7)
    {
        number_bands = atoi(argv[1]);
        nvband = atoi(argv[2]);
        ncouls = atoi(argv[3]);
        ngpown = atoi(argv[4]);
        nFreq = atoi(argv[5]);
        nfreqeval = atoi(argv[6]);
    }
    else
    {
        cout << "Incorrect Parameters!!! The correct form is " << endl;
        cout << "./a.out number_bands nvband ncouls ngpown nFreq nfreqeval " << endl;
        exit(0);
    }


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
        start_achsDtemp_Kernel, end_achsDtemp_Kernel, \
        start_asxDtemp_Kernel, end_asxDtemp_Kernel, \
        start_achDtemp_Kernel, end_achDtemp_Kernel,
        start_achDtemp_cor_Kernel, end_achDtemp_cor_Kernel, \
        start_preKernel, end_preKernel;

    gettimeofday(&start_preKernel, NULL);

    //OpenMP variables
    int tid = 0, numThreads = 0;
//#pragma omp parallel shared(numThreads) private(tid)
//    {
//        tid = omp_get_thread_num();
//        if(tid == 0)
//            numThreads = omp_get_num_threads();
//    }

    cout << "Number of Threads = " << numThreads << \
        "\n number_bands = " << number_bands << \
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
    long double mem_alloc = 0.00;

    CustomComplex<double> *aqsntemp = new CustomComplex<double>[number_bands * ncouls];
    mem_alloc += (number_bands * ncouls * sizeof(CustomComplex<double>));

    CustomComplex<double> *aqsmtemp= new CustomComplex<double>[number_bands * ncouls];
    mem_alloc += (number_bands * ncouls * sizeof(CustomComplex<double>));

    CustomComplex<double> *I_epsR_array = new CustomComplex<double>[nFreq * ngpown * ncouls];
    mem_alloc += (nFreq * ngpown * ncouls * sizeof(CustomComplex<double>));

    //Variables used :
    CustomComplex<double> achsDtemp(0.00, 0.00);

    /************************************************************************************/
    //Initialize the data structures
    for(int ig = 0; ig < ngpown; ++ig)
        inv_igp_index[ig] = ig;

    for(int ig = 0; ig < ncouls; ++ig)
        indinv[ig] = ig;

    for(int i=0; i<number_bands; ++i)
    {
        for(int j=0; j<ncouls; ++j)
        {
            aqsmtemp[i*ncouls+j] = expr;
            aqsntemp[i*ncouls+j] = expr;
        }
    }

    for(int i=0; i<nFreq; ++i)
    {
        for(int j=0; j<ngpown; ++j)
        {
            for(int k=0; k<ncouls; ++k)
            {
                I_epsR_array[i*ngpown*ncouls + j * ncouls + k] = expR;
            }
        }
    }

    for(int i=0; i<ncouls; ++i)
        vcoul[i] = 1.00;

    achsDtemp_Kernel(number_bands, ngpown, ncouls, inv_igp_index, indinv, \
            aqsntemp, aqsmtemp, vcoul, achsDtemp, I_epsR_array);

//Free the allocated memory
    delete[] aqsntemp;
    delete[] aqsmtemp;
    delete[] inv_igp_index;
    delete[] indinv;
    delete[] vcoul;

    return 0;
}
