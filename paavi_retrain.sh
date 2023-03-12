#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=42
#SBATCH --mem-per-cpu=3700
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:quadro_rtx_6000:3
#SBATCH --partition=gpu

module purge
module load GCC/10.2.0 parallel/20210322 Python/3.8.6 
pip install -e src/


MY_PARALLEL_OPTS="-N 1 --delay .2 -j $SLURM_NTASKS --joblog parallel-${SLURM_JOBID}.log"
MY_SRUN_OPTS="-N 1 -n 1 --exclusive"
MY_EXEC="echo running job {1} & python src/retrain.py --job_no {1} --attr_model ./maskedDQN_final.zip --basic_model ./maskedDQN_final.zip"

parallel $MY_PARALLEL_OPTS srun $MY_SRUN_OPTS $MY_EXEC ::: {0..124}