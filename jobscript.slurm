#!/bin/bash --login                                                                                                     
#SBATCH -J [INSERT JOB NAME HERE]                                                                                                    
#SBATCH -p gpuA40GB                                                                                                     
#SBATCH -G 2                                                                                                            
#SBATCH -n 24                                                                                                           
#SBATCH -t 01:00:00                                                                                                     

module purge
module load libs/cuda

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $≈CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)."

uv run python [INSERT SCRIPT NAME HERE]

