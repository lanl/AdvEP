#!/bin/bash

#SBATCH --job-name=CEL_BPTTSteps
#SBATCH --partition=shared-gpu-ampere
##SBATCH --constraint gpu_count:1
#SBATCH --qos long
#SBATCH --time=1:30:00

conda init bash
source ~/.bashrc
conda activate EPEnv

if [ $1 -eq 11 ]; then 
    python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act my_hard_sig --todo attack --T1 250 --T2 30 --mbs 128 --alg BPTT --loss cel --softmax --save --device 0 --seed 11 --attack-step $2 --predict-step $3 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BPTT/cel/2023-06-19/17-19-25_gpu0/
elif [ $1 -eq 13 ]; then 
    python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act my_hard_sig --todo attack --T1 250 --T2 30 --mbs 128 --alg BPTT --loss cel --softmax --save --device 0 --seed 13 --attack-step $2 --predict-step $3 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BPTT/cel/2023-06-19/18-28-16_gpu0/
elif [ $1 -eq 17 ]; then
    python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act my_hard_sig --todo attack --T1 250 --T2 30 --mbs 128 --alg BPTT --loss cel --softmax --save --device 0 --seed 17 --attack-step $2 --predict-step $3 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BPTT/cel/2023-06-19/19-10-16_gpu0/
elif [ $1 -eq 19 ]; then 
    python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act my_hard_sig --todo attack --T1 250 --T2 30 --mbs 128 --alg BPTT --loss cel --softmax --save --device 0 --seed 19 --attack-step $2 --predict-step $3 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BPTT/cel/2023-06-19/19-16-24_gpu0/
else
    python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act my_hard_sig --todo attack --T1 250 --T2 30 --mbs 128 --alg BPTT --loss cel --softmax --save --device 0 --seed 31 --attack-step $2 --predict-step $3 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BPTT/cel/2023-06-19/17-17-24_gpu0/
fi