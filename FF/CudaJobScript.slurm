#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:20:00
#SBATCH --job-name=ff-cuda-job
#SBATCH --exclusive=user

#Job Script for ff-CUDA on google cloud
nvprof1="nvprof -o ff_time.nvprof"
nvprof2="nvprof --analysis-metrics -o ff_metrics.nvprof"
exec=ff_GPUComplex
input="15023 1998 33401 66 15 10"
pwd; hostname; date
./${exec}.ex ${input}
nvprof --kernels "achsDtemp_solver_1D" --metrics all ./${exec}.ex ${input}
#nvprof --kernels "asxDtemp_solver_1D" --kernels "achDtemp_cor_solver_1D" --metrics flop_count_dp --metrics dram_read_transactions --metrics dram_write_transactions ./${exec}.ex ${input}
#nvprof --kernels "achsDtemp_solver_2D" --metrics flop_count_dp flop_count_dp ./${exec}.ex ${input}
date
