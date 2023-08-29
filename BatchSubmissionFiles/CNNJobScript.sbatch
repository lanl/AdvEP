#!/bin/bash -l
#SBATCH --job-name=adv_image_gen
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --partition=shared-gpu-ampere


conda activate CNNEnv
python adversarial_CIFAR_CNNEnv.py $1 $2 $3
