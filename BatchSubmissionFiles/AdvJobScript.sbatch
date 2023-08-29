#!/bin/bash -l
#SBATCH --job-name=adv_image_gen
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --partition=shared-gpu-ampere


conda activate LEP
eps=(0.05 0.1 0.25 0.5 0.75 1.0)
python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 40 --act my_hard_sig --todo advtrain --T1 250 --T2 30 --mbs 128 --alg EP --thirdphase --betas 0.0 0.5 --loss cel --softmax --device 0 --eps ${eps[$1]}

