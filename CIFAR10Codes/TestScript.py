# import necessary libraries
import torch
def adder(A,t):
    g = 0
    g += 2
    f = [A]
    return g*f[0]
# define a tensor
A = torch.tensor(5., requires_grad=True)
B = torch.tensor(5.,requires_grad=True)
print("Tensor-A:", A)

# define a function using above defined
# tensor
x = A**3
x = x+ B**2
phi = adder(x,3)
print("x:", x)

# call the backward method
phi.backward()

# print the gradient using .grad
print("A.grad:", A.grad)
print("B.grad",B.grad)
python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 
--kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd 
--lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 
--act my_hard_sig --todo evaluate --T1 250 --T2 30 --mbs 128 --alg EP --random-sign --betas 0.0 0.5 
--loss mse --device 0 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/mse/2023-05-22/12-28-59_gpu0


python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 
--kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd 
--lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 80 
--act my_hard_sig --todo evaluate --T1 250 --T2 30 --mbs 128 --alg EP --thirdphase --betas 0.0 0.5 
--loss mse --device 0 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/mse/2023-05-22/12-31-48_gpu0/

python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 
--kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd 
--lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 120 
--act my_hard_sig --todo evaluate --T1 250 --T2 30 --mbs 128 --alg EP --thirdphase --betas 0.0 0.5 
--loss cel --device 0 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/cel/2023-05-23/15-40-03_gpu0/

python main.py --model CNN --task CIFAR10 --data-aug --channels 128 256 512 512 
--kernels 3 3 3 3 --pools mmmm --strides 1 1 1 1 --paddings 1 1 1 0 --fc 10 --optim sgd 
--lrs 0.25 0.15 0.1 0.08 0.05 --wds 3e-4 3e-4 3e-4 3e-4 3e-4 --mmt 0.9 --lr-decay --epochs 40 
--act my_hard_sig --todo evaluate --T1 250 --T2 30 --mbs 128 --alg EP --thirdphase --betas 0.0 0.5 
--loss cel --softmax --device 0 --load-path /vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/cel/2023-05-23/16-52-21_gpu0/
