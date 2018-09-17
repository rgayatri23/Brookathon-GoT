#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --job-name=ff-OpenACC-job
#SBATCH --exclusive=user

#Job Script for ff-OpenACC on google cloud
exec=ff
input="15023 1998 33401 66 15 10"
pwd; hostname; date
./${exec}.ex ${input}
#pgprof -o ff-pgprof.nvvp ./${exec}.ex ${input}
date
