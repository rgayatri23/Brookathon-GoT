#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --job-name=ff-cuda-job
#SBATCH --exclusive=user

#Job Script for ff-CUDA on google cloud
nvprof1="nvprof -o ff_time.nvprof"
nvprof2="nvprof --kernels ::achsDtemp_solver_2D_v2: --analysis-metrics -o ff_metrics.nvprof"
exec=ff_GPUComplex
input="15023 1998 33401 66 15 10"
pwd; hostname; date
rm -r ff_metrics.nvprof
./${exec}.ex ${input}
#${nvprof1} ./${exec}.ex ${input}
#${nvprof2} ./${exec}.ex ${input}
date
