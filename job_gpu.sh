#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:quadro_rtx_6000:7
#SBATCH --partition=gpu

module purge
module load GCC/10.2.0  CUDA/11.1.1  OpenMPI/4.0.5
module load 
pip install 

echo 'running Phase1.py'
srun python retrain.py --num_cpus 48 --num_gpus 7
