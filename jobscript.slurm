#!/bin/bash --login                                                                                                     
#SBATCH -J COMBSCRPT                                                                                                    
#SBATCH -p gpuA40GB                                                                                                     
#SBATCH -G 2                                                                                                            
#SBATCH -n 24                                                                                                           
#SBATCH -t 01:00:00                                                                                                     

module load libs/cuda/12.4.1

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $≈CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)."

uv run python [INSERT SCRIPT NAME HERE]
$SLURM_JOB_ID

