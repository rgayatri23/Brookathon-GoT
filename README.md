# Brookathon-GoT
GPP and FF mini-apps from Berkeley GW application suite.
Complex number data type from CustomComplex and GPUComplex class available in ComplexClass dir. 


Master branch uses CustomComplex class to represent a complex number. This is templated class which allows one to choose the data type to represent the real and imaginary parts of the complex number. 

CUDA branch uses GPUComplex, a modified version of CustomClass which inherits from the double2 vector type provided by CUDA framework.


Problem sizes:
GPP : ./gpp.ex 512 2 32768 20
    Output : 
        Final achtemp
        ( -264241152.090614, 1321205760.021389)

FF :  ./ff.ex 15023 1998 33401 66 15 10 
    Output : 
        achsDtemp = ( -4139711589.750000, -4139711589.750000)
        asxDtemp = ( 1101130767.000000, -1089006204.000000)
        achDtemp_cor = ( 0.000000, -1095068485.500000)

Runtime numbers : 
-----------|-------|-------|
T[seconds] | FF    | GPP   |
-----------|-------|-------|
Haswell    | 58.83 | 7.65  |
-----------|-------|-------|
KNL        | 18.5  | 3.2   |
-----------|-------|-------|
Powe9      | 50.5  | 27.5  |
-----------|-------|-------|
Volta(acc) | 6.5   | 3.5   |
-----------|-------|-------|
Volta(cuda)| 2.1   | 0.8   |
-----------|-------|-------|
