#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:20:00
#SBATCH --job-name=ff-OpenACC-job
#SBATCH --exclusive=user

#Job Script for ff-OpenACC on google cloud
exec=ff
input="15023 1998 33401 66 15 10"
pwd; hostname; date
#export PGI_ACC_TIME=1
./${exec}.ex ${input}
#export PGI_ACC_SYNCHRONUS=1
#pgprof -o OpenACC-ff-pgprof.out ./${exec}.ex ${input}
#nvprof --metrics all ./${exec}.ex ${input}
#nvprof --kernels "_Z16achsDtemp_KerneliiiiPiS_P13CustomComplexIdES2_S2_S2_PdRS1_Rd" --metrics all ./${exec}.ex ${input}

#cuda-memcheck error-exitcode ./${exec}.ex ${input}
#pgprof -o ff-pgprof.nvvp ./${exec}.ex ${input}
date
