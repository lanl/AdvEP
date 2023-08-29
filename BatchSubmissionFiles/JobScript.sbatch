#!/bin/bash -l
#SBATCH --job-name=adv_image_gen
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --partition=shared-gpu-ampere

conda activate LEP

python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 80 --act my_hard_sig --todo attack --T1 250 --T2 30 --mbs 128 --alg EP --random-sign --betas 0.0 0.5 --loss mse --save --device 0 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/mse/2023-05-22/12-28-59_gpu0/

python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 40 --act my_hard_sig.9 --lr-decay --epochs 80 --act my_hard_sig --todo attack --T1 250 --T2 30 --mbs 128 --alg EP --thirdphase --betas 0.0 0.5 --loss mse --device 0 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/mse/2023-05-22/12-31-48_gpu0/

python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 40 --act my_hard_sig --todo attack --T1 250 --T2 30 --mbs 128 --alg EP --thirdphase --betas 0.0 0.5 --loss cel --softmax --device 0 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/cel/2023-05-23/16-52-21_gpu0/

python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 80 --act my_hard_sig --todo attack --T1 250 --T2 30 --mbs 128 --alg BPTT --loss mse  --device 0 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BPTT/mse/2023-05-22/13-52-18_gpu0/

python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 --kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd --lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 --act my_hard_sig --todo attack --T1 250 --T2 25 --mbs 128 --alg BPTT --loss cel --softmax --save --device 0 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BPTT/cel/2023-05-30/15-31-49_gpu0/

