#!/bin/bash

#SBATCH --job-name=CEL_SymmetricSteps
#SBATCH --partition=shared-gpu-ampere
##SBATCH --constraint gpu_count:1
#SBATCH --qos long
#SBATCH --time=10:00:00

conda init bash
source ~/.bashrc
conda activate LEP

# EP with symmetric gradient estimate
python main.py --model 'CNN' --task 'CIFAR10' --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools 'mmmm' --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim 'sgd' --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act 'my_hard_sig' --todo 'train' --T1 250 --T2 25 --mbs 128 --alg 'EP' --betas 0.0 1.0 --thirdphase --loss 'cel' --save --device 0  --seed $1
