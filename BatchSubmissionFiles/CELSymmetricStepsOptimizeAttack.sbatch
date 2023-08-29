#!/bin/bash

#SBATCH --job-name=CEL_SymmetricSteps
#SBATCH --partition=shared-gpu-ampere
##SBATCH --constraint gpu_count:1
#SBATCH --qos long
#SBATCH --time=2:30:00

conda init bash
source ~/.bashrc
conda activate EPEnv

if [ $1 -eq 11 ]; then
    python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act my_hard_sig --todo attack --T1 60 --T2 10 --mbs 128 --alg EP --thirdphase --betas 0.0 1.5 --loss cel --softmax --device 0 --save --seed 11 --attack-step 70 --predict-step 60 --attack-norm 2 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/cel/2023-07-07/10-56-31_gpu0/
elif [ $1 -eq 13 ]; then 
    python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act my_hard_sig --todo attack --T1 60 --T2 4 --mbs 128 --alg EP --thirdphase --betas 0.0 0.5 --loss cel --softmax --device 0 --save --seed 11 --attack-step 60 --predict-step 60 --attack-norm 2 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/cel/2023-07-09/12-26-07_gpu0/
elif [ $1 -eq 17 ]; then 
    python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act my_hard_sig --todo attack --T1 55 --T2 15 --mbs 128 --alg EP --thirdphase --betas 0.0 1.0 --loss cel --softmax --device 0 --save --seed 11 --attack-step 55 --predict-step 55 --attack-norm 2 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/cel/2023-07-10/10-46-00_gpu0/
elif [ $1 -eq 19 ]; then
    python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act my_hard_sig --todo attack --T1 250 --T2 30 --mbs 128 --alg EP --thirdphase --betas 0.0 0.5 --loss cel --softmax --device 0 --save --seed 19 --attack-step $2 --predict-step $3 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/cel/2023-06-14/17-06-24_gpu0/
else 
    python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act my_hard_sig --todo attack --T1 250 --T2 30 --mbs 128 --alg EP --thirdphase --betas 0.0 0.5 --loss cel --softmax --device 0 --save --seed 31 --attack-step $2 --predict-step $3 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/cel/2023-06-15/00-48-27_gpu0/
fi