import torch
import numpy as np
from torch import tensor
from numpy import array

x = tensor(1., requires_grad=True)
w = torch.nn.Conv2d(3,128,kernel_size=(3,3),padding=1)
x = torch.randn(120,3,32,32)
print('x:', x.shape)
print(w(x).shape)
z = w(x)
w2 = torch.nn.Conv2d(128,128,kernel_size=(3,3),padding=1)
z2=w2(z)
y = (z2*2).sum()
print(y)
y.backward()
print(w.weight.grad.size())
