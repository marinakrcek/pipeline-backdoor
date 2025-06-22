#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/home/jtelintelo/pipeline-backdoor/slurm/output/%j-%x.out
#SBATCH --error=/home/jtelintelo/pipeline-backdoor/slurm/error/%j-%x.err

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source venv/bin/activate

python clean_tinystories-8M.py

deactivate