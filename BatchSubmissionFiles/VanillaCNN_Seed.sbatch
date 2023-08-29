#!/bin/bash

#SBATCH --job-name=CEL_VCNN
#SBATCH --partition=shared-gpu
##SBATCH --constraint gpu_count:1
#SBATCH --qos long
#SBATCH --time=4:00:00

conda init bash
source ~/.bashrc
conda activate CNNEnv

python adversarial_CIFAR_CNNEnv.py $1