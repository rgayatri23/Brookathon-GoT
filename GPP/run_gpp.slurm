#!/bin/bash
#SBATCH -N 1
#SBATCH -A m1759
#SBATCH -S 4
#SBATCH -J gppKerKokkos_KNL
#SBATCH -p debug
#SBATCH -t 00:30:00 
#SBATCH -C knl 
#SBATCH -L SCRATCH

#module use /global/common/cori/software/modulefiles
#module unload advisor
#module load advisor/2018.integrated_roofline
#module load sde
#module load vtune
#module load likwid

#ARCH=hsw
ARCH=knl

export OMP_PLACES=threads
export OMP_PROC_BIND=spread

EXEC=gpp

Input="512 2 32768 20"

num_mpi_ranks=1
numHT=1
threads_per_rank=1
numLC=1

if [ ${ARCH} = "knl" ]; then
    numHT=4
    threads_per_rank=$((64/${num_mpi_ranks}))
    numLC=$(( ${threads_per_rank}*${numHT} ))
    srun_command="srun -n${num_mpi_ranks} -c${numLC} --cpu_bind=cores"
fi

if [ ${ARCH} = "hsw" ]; then
    numHT=2
    threads_per_rank=$((32/${num_mpi_ranks}))
    numLC=$(( ${threads_per_rank}*${numHT} ))
    srun_command="srun -n${num_mpi_ranks} -c${numLC} --cpu_bind=cores"
fi

#export OMP_NUM_THREADS=${threads_per_rank}
export OMP_NUM_THREADS=${numLC}
echo "OMP_NUM_THREADS="${OMP_NUM_THREADS}

echo "${srun_command} ./${EXEC}.ex ${Input}"

${srun_command} ./${EXEC}.ex ${Input} 
