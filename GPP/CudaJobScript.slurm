#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --job-name=gpp-cuda-job

#Job Script for gpp-CUDA on google cloud
exec=gpp_GPUComplex
input="512 2 32768 20"
pwd; hostname; date
./${exec}.ex ${input}
date
