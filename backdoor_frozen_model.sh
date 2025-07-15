#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/jtelintelo/pipeline-backdoor/slurm/output/%j-%x.out
#SBATCH --error=/home/jtelintelo/pipeline-backdoor/slurm/error/%j-%x.err

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source venv/bin/activate

python backdoor_frozen_model.py

deactivate