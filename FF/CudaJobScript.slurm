#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --job-name=ff-cuda-job

#Job Script for ff-CUDA on google cloud
exec=ff_GPUComplex
input="15023 1998 33401 66 15 10"
pwd; hostname; date
./${exec}.ex ${input}
date
