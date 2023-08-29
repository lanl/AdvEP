import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F

import os
from datetime import datetime
import time
import math
from data_utils import *

from itertools import repeat
from torch.nn.parameter import Parameter
import collections
import matplotlib
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch
from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.estimators.classification import PyTorchClassifier
from art.estimators.classification import BlackBoxClassifierNeuralNetwork
from art.attacks.evasion.square_attack import SquareAttack
from art.attacks.evasion.auto_attack import AutoAttack
matplotlib.use('Agg')
characteristic_time = []
characteristic_param = 0
attack_param = 0
# Activation functions
def my_sigmoid(x):
    return 1/(1+torch.exp(-4*(x-0.5)))

def hard_sigmoid(x):
    return (1+F.hardtanh(2*x-1))*0.5

def ctrd_hard_sig(x):
    return (F.hardtanh(2*x))*0.5

def my_hard_sig(x):
    return (1+F.hardtanh(x-1))*0.5





# Some helper functions
def grad_or_zero(x):
    if x.grad is None:
        return torch.zeros_like(x).to(x.device)
    else:
        return x.grad

def neurons_zero_grad(neurons):
    for idx in range(len(neurons)):
        if neurons[idx].grad is not None:
            neurons[idx].grad.zero_()

def copy(neurons):
    copy = []
    for n in neurons:
        copy.append(torch.empty_like(n).copy_(n.data).requires_grad_())
    return copy






def make_pools(letters):
    pools = []
    for p in range(len(letters)):
        if letters[p]=='m':
            pools.append( torch.nn.MaxPool2d(2, stride=2) )
        elif letters[p]=='a':
            pools.append( torch.nn.AvgPool2d(2, stride=2) )
        elif letters[p]=='i':
            pools.append( torch.nn.Identity() )
    return pools
        


       
def my_init(scale):
    def my_scaled_init(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))
            m.weight.data.mul_(scale)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
                m.bias.data.mul_(scale)
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, math.sqrt(5))
            m.weight.data.mul_(scale)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
                m.bias.data.mul_(scale)
    return my_scaled_init




        
# Multi-Layer Perceptron

class P_MLP(torch.nn.Module):
    def __init__(self, archi, activation=torch.tanh):
        super(P_MLP, self).__init__()
        
        self.activation = activation
        self.archi = archi
        self.softmax = False    #Softmax readout is only defined for CNN and VFCNN       
        self.nc = self.archi[-1]

        # Synapses
        self.synapses = torch.nn.ModuleList()
        for idx in range(len(archi)-1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx+1], bias=True))

            
    def Phi(self, x, y, neurons, beta, criterion):
        #Computes the primitive function given static input x, label y, neurons is the sequence of hidden layers neurons
        # criterion is the loss 
        x = x.view(x.size(0),-1) # flattening the input
        
        layers = [x] + neurons  # concatenate the input to other layers
        
        # Primitive function computation
        phi = 0.0
        for idx in range(len(self.synapses)):
            phi += torch.sum( self.synapses[idx](layers[idx]) * layers[idx+1], dim=1).squeeze() # Scalar product s_n.W.s_n-1
        
        if beta!=0.0: # Nudging the output layer when beta is non zero 
            if criterion.__class__.__name__.find('MSE')!=-1:
                y = F.one_hot(y, num_classes=self.nc)
                L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
            else:
                L = criterion(layers[-1].float(), y).squeeze()     
            phi -= beta*L
        
        return phi
    
    
    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
        # Run T steps of the dynamics for static input x, label y, neurons and nudging factor beta.
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phi = self.Phi(x, y, neurons, beta, criterion) # Computing Phi
            init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True) #Initializing gradients
            grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=check_thm) # dPhi/ds

            for idx in range(len(neurons)-1):
                neurons[idx] = self.activation( grads[idx] )  # s_(t+1) = sigma( dPhi/ds )
                if check_thm:
                    neurons[idx].retain_grad()
                else:
                    neurons[idx].requires_grad = True
             
            if not_mse:
                neurons[-1] = grads[-1]
            else:
                neurons[-1] = self.activation( grads[-1] )

            if check_thm:
                neurons[-1].retain_grad()
            else:
                neurons[-1].requires_grad = True

        return neurons


    def init_neurons(self, mbs, device):
        # Initializing the neurons
        neurons = []
        append = neurons.append
        for size in self.archi[1:]:  
            append(torch.zeros((mbs, size), requires_grad=True, device=device))
        return neurons


    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        # Computing the EP update given two steady states neurons_1 and neurons_2, static input x, label y
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        phi_1 = phi_1.mean()
        
        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()
        
        delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)        
        delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem
        

         
# Vector Field Multi-Layer Perceptron

class VF_MLP(torch.nn.Module):
    def __init__(self, archi, activation=torch.tanh):
        super(VF_MLP, self).__init__()
        
        self.activation = activation
        self.archi = archi
        self.softmax = False        
        self.nc = self.archi[-1]

        # Forward synapses
        self.synapses = torch.nn.ModuleList()
        for idx in range(len(archi)-1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx+1], bias=True))

        # Backward synapses
        self.B_syn = torch.nn.ModuleList()
        for idx in range(1, len(archi)-1):            
            self.B_syn.append(torch.nn.Linear(archi[idx+1], archi[idx], bias=False))


    def Phi(self, x, y, neurons, beta, criterion, neurons_2=None):
        # For assymetric connections each layer has its own phi 
        x = x.view(x.size(0),-1)
        
        layers = [x] + neurons
        
        phis = []

        if neurons_2 is None:  # Standard case for the dynamics
            for idx in range(len(self.synapses)-1):
                phi = torch.sum( self.synapses[idx](layers[idx]) * layers[idx+1], dim=1).squeeze()
                phi += torch.sum( self.B_syn[idx](layers[idx+2]) * layers[idx+1], dim=1).squeeze()
                phis.append(phi)        

            phi = torch.sum( self.synapses[-1](layers[-2]) * layers[-1], dim=1).squeeze()
            if beta!=0.0:
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
                else:
                    L = criterion(layers[-1].float(), y).squeeze()     
                phi -= beta*L
        
            phis.append(phi)
        
        else:  # Used only for computing the vector field EP update
            layers_2 = [x] + neurons_2
            for idx in range(len(self.synapses)-1):
                phi = torch.sum( self.synapses[idx](layers[idx]) * layers_2[idx+1], dim=1).squeeze()
                phi += torch.sum( self.B_syn[idx](layers[idx+2]) * layers_2[idx+1], dim=1).squeeze()
                phis.append(phi)        

            phi = torch.sum( self.synapses[-1](layers[-2]) * layers_2[-1], dim=1).squeeze()
            if beta!=0.0:
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5*criterion(layers_2[-1].float(), y.float()).sum(dim=1).squeeze()   
                else:
                    L = criterion(layers_2[-1].float(), y).squeeze()     
                phi -= beta*L
        
            phis.append(phi)

        return phis
    
    
    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
        
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device

        for t in range(T):
            phis = self.Phi(x, y, neurons, beta, criterion)
            for idx in range(len(neurons)-1):
                init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                grad = torch.autograd.grad(phis[idx], neurons[idx], grad_outputs=init_grad, create_graph=check_thm)

                neurons[idx] = self.activation( grad[0] )
                if check_thm:
                    neurons[idx].retain_grad()
                else:
                    neurons[idx].requires_grad = True
             
            init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
            grad = torch.autograd.grad(phis[-1], neurons[-1], grad_outputs=init_grad, create_graph=check_thm)
            if not_mse:
                neurons[-1] = grad[0]
            else:
                neurons[-1] = self.activation( grad[0] )

            if check_thm:
                neurons[-1].retain_grad()
            else:
                neurons[-1].requires_grad = True

        return neurons


    def init_neurons(self, mbs, device):
        
        neurons = []
        append = neurons.append
        for size in self.archi[1:]:  
            append(torch.zeros((mbs, size), requires_grad=True, device=device))
        return neurons


    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            phis_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phis_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        
        phis_2 = self.Phi(x, y, neurons_1, beta_2, criterion, neurons_2=neurons_2)
     
        for idx in range(len(neurons_1)):
            phi_1 = phis_1[idx].mean()
            phi_2 = phis_2[idx].mean()
            delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)        
            delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1)
        

   
    
    
# Convolutional Neural Network

class P_CNN(torch.nn.Module):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=hard_sigmoid, softmax=False):
        super(P_CNN, self).__init__()

        # Dimensions used to initialize neurons
        self.in_size = in_size
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.fc = fc
        self.nc = fc[-1]        

        self.activation = activation
        self.pools = pools
        
        self.synapses = torch.nn.ModuleList()
        
        self.softmax = softmax # whether to use softmax readout or not

        size = in_size # size of the input : 32 for cifar10

        for idx in range(len(channels)-1): 
            self.synapses.append(torch.nn.Conv2d(channels[idx], channels[idx+1], kernels[idx], 
                                                 stride=strides[idx], padding=paddings[idx], bias=True))
                
            size = int( (size + 2*paddings[idx] - kernels[idx])/strides[idx] + 1 )          # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - pools[idx].kernel_size)/pools[idx].stride + 1 )   # size after Pool

        size = size * size * channels[-1]        
        fc_layers = [size] + fc

        for idx in range(len(fc)):
            self.synapses.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx+1], bias=True))
        
    def init_neurons(self, mbs, device):
        
        neurons = []
        append = neurons.append
        size = self.in_size
        for idx in range(len(self.channels)-1): 
            size = int( (size + 2*self.paddings[idx] - self.kernels[idx])/self.strides[idx] + 1 )   # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - self.pools[idx].kernel_size)/self.pools[idx].stride + 1 )  # size after Pool
            append(torch.zeros((mbs, self.channels[idx+1], size, size), requires_grad=True, device=device))

        size = size * size * self.channels[-1]
        
        if not self.softmax:
            for idx in range(len(self.fc)):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))
        else:
            # we *REMOVE* the output layer from the system
            for idx in range(len(self.fc) - 1):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))            
            
        return neurons

    def Phi(self, x, y, neurons, beta, criterion):

        mbs = x.size(0)       
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)

        layers = [x] + neurons
        phi = 0#torch.zeros(x.shape[0], device=x.device, requires_grad=True)

        #Phi computation changes depending on softmax == True or not
        if not self.softmax:
            for idx in range(conv_len):    
                phi = phi + torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
            for idx in range(conv_len, tot_len):
                phi = phi + torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
             
            if beta!=0.0:
                if criterion.__class__.__name__.find('MSE')!=-1:
                    y = F.one_hot(y, num_classes=self.nc)
                    L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
                else:
                    L = criterion(layers[-1].float(), y).squeeze()             
                phi = phi - beta*L

        else:
            # the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
            for idx in range(conv_len):
                phi = phi + torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
            for idx in range(conv_len, tot_len-1):
                phi = phi + torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
             
            # the prediction is made with softmax[last weights[penultimate layer]]
            if beta!=0.0:
                L = criterion(self.synapses[-1](layers[-1].view(mbs,-1)).float(), y).squeeze()             
                phi = phi - beta*L            
        
        return phi
    

    def forward(self, x, y=0, neurons=None, T=29, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
        global characteristic_param, characteristic_time, attack_param
        if(attack_param == 1):
            neurons = self.init_neurons(x.size(0), x.device)
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device    
        for idx in range(len(self.pools)):
            self.pools[idx].return_indices = False

        if check_thm:
            for t in range(T):
                phi = self.Phi(x, y, neurons, beta, criterion)
                init_grads = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=True)

                for idx in range(len(neurons)-1):
                    neurons[idx] = self.activation( grads[idx] )
                    neurons[idx].retain_grad()
             
                if not_mse and not(self.softmax):
                    neurons[-1] = grads[-1]
                else:
                    neurons[-1] = self.activation( grads[-1] )

                neurons[-1].retain_grad()
        else:
             for t in range(T):
                phi = self.Phi(x, y, neurons, beta, criterion)
                init_grads = torch.ones(mbs, dtype=torch.float, device=device, requires_grad=True)
                if mbs != 1:
                    grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=False)
                else:
                    grads = torch.autograd.grad(phi, neurons, create_graph=False)
                
                    
                for idx in range(len(neurons)-1):
                    neurons[idx] = self.activation( grads[idx] )
                    neurons[idx].requires_grad = True
                    
             
                if not_mse and not(self.softmax):
                    neurons[-1] = grads[-1]
                else:
                    neurons[-1] = self.activation( grads[-1] )

                neurons[-1].requires_grad = True
                
                
                if(characteristic_param):
                    if not self.softmax:
                        characteristic_time.append(neurons[-1])  # in this cas prediction is done directly on the last (output) layer of neurons
                    else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
                        characteristic_time.append(F.softmax(self.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1))
                   
                    

        return neurons
    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        phi_1 = phi_1.mean()
        
        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()
        
        delta_phi = (phi_2 - phi_1)/(beta_1 - beta_2)        
        delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem


   
 
# Vector Field Convolutional Neural Network

class VF_CNN(torch.nn.Module):
    def __init__(self, in_size, channels, kernels, strides, fc, pools, paddings, activation=hard_sigmoid, softmax = False, same_update=False):
        super(VF_CNN, self).__init__()

        # Dimensions used to initialize neurons
        self.in_size = in_size
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.fc = fc
        self.nc = fc[-1]        

        self.activation = activation
        self.pools = pools
        
        self.synapses = torch.nn.ModuleList()  # forward connections
        self.B_syn = torch.nn.ModuleList()     # backward connections

        self.same_update = same_update         # whether to use the same update for forward ans backward connections
        self.softmax = softmax                 # whether to use softmax readout


        size = in_size

        for idx in range(len(channels)-1): 
            self.synapses.append(torch.nn.Conv2d(channels[idx], channels[idx+1], kernels[idx], 
                                                 stride=strides[idx], padding=paddings[idx], bias=True))
                
            if idx>0:  # backward synapses except for first layer
                self.B_syn.append(torch.nn.Conv2d(channels[idx], channels[idx+1], kernels[idx],
                                                      stride=strides[idx], padding=paddings[idx], bias=False))

            size = int( (size + 2*paddings[idx] - kernels[idx])/strides[idx] + 1 )          # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - pools[idx].kernel_size)/pools[idx].stride + 1 )   # size after Pool

        size = size * size * channels[-1]
        
        fc_layers = [size] + fc

        for idx in range(len(fc)):
            self.synapses.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx+1], bias=True))
            if not(self.softmax and (idx==(len(fc)-1))):
                self.B_syn.append(torch.nn.Linear(fc_layers[idx], fc_layers[idx+1], bias=False))


    def angle(self): # computes the angle between forward and backward weights
        angles = []
        with torch.no_grad():
            for idx in range(len(self.B_syn)):
                fnorm = self.synapses[idx+1].weight.data.pow(2).sum().pow(0.5).item()
                bnorm = self.B_syn[idx].weight.data.pow(2).sum().pow(0.5).item()
                cos = self.B_syn[idx].weight.data.mul(self.synapses[idx+1].weight.data).sum().div(fnorm*bnorm)
                angle = torch.acos(cos).item()*(180/(math.pi))
                angles.append(angle)                
        return angles


    def Phi(self, x, y, neurons, beta, criterion, neurons_2=None):

        mbs = x.size(0)       
        conv_len = len(self.kernels)
        tot_len = len(self.synapses)
        bck_len = len(self.B_syn)

        layers = [x] + neurons        
        phis = []

        if neurons_2 is None: #neurons 2 is not None only when computing the EP update for different updates between forward and backward
            #Phi computation changes depending on softmax == True or not
            if not self.softmax:

                for idx in range(conv_len-1):    
                    phi = torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
                    phi += torch.sum( self.pools[idx+1](self.B_syn[idx](layers[idx+1])) * layers[idx+2], dim=(1,2,3)).squeeze()
                    phis.append(phi)

                phi = torch.sum( self.pools[conv_len-1](self.synapses[conv_len-1](layers[conv_len-1])) * layers[conv_len], dim=(1,2,3)).squeeze()
                phi += torch.sum( self.B_syn[conv_len-1](layers[conv_len].view(mbs,-1)) * layers[conv_len+1], dim=1).squeeze()            
                phis.append(phi)            

                for idx in range(conv_len+1, tot_len-1):
                    phi = torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
                    phi += torch.sum( self.B_syn[idx](layers[idx+1].view(mbs,-1)) * layers[idx+2], dim=1).squeeze()             
                    phis.append(phi)

                phi = torch.sum( self.synapses[-1](layers[-2].view(mbs,-1)) * layers[-1], dim=1).squeeze()
                if beta!=0.0:
                    if criterion.__class__.__name__.find('MSE')!=-1:
                        y = F.one_hot(y, num_classes=self.nc)
                        L = 0.5*criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()   
                    else:
                        L = criterion(layers[-1].float(), y).squeeze()             
                    phi -= beta*L
                phis.append(phi)

            else:
                #the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
                for idx in range(conv_len-1):    
                    phi = torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers[idx+1], dim=(1,2,3)).squeeze()     
                    phi += torch.sum( self.pools[idx+1](self.B_syn[idx](layers[idx+1])) * layers[idx+2], dim=(1,2,3)).squeeze()
                    phis.append(phi)
            
                phi = torch.sum( self.pools[conv_len-1](self.synapses[conv_len-1](layers[conv_len-1])) * layers[conv_len], dim=(1,2,3)).squeeze()
                if bck_len>=conv_len:
                    phi += torch.sum( self.B_syn[conv_len-1](layers[conv_len].view(mbs,-1)) * layers[conv_len+1], dim=1).squeeze()            
                    phis.append(phi)            

                    for idx in range(conv_len, tot_len-2):
                        phi = torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers[idx+1], dim=1).squeeze()
                        phi += torch.sum( self.B_syn[idx](layers[idx+1].view(mbs,-1)) * layers[idx+2], dim=1).squeeze()             
                        phis.append(phi)
                    
                    phi = torch.sum( self.synapses[-2](layers[-2].view(mbs,-1)) * layers[-1], dim=1).squeeze()
                    if beta!=0.0:
                        L = criterion(self.synapses[-1](layers[-1].view(mbs,-1)).float(), y).squeeze()             
                        phi -= beta*L
                    phis.append(phi)                    

                #the prediction is made with softmax[last weights[penultimate layer]]
                elif beta!=0.0:
                    L = criterion(self.synapses[-1](layers[-1].view(mbs,-1)).float(), y).squeeze()             
                    phi -= beta*L
                    phis.append(phi)            
                else:
                    phis.append(phi)
           
        else:
            layers_2 = [x] + neurons_2
            #Phi computation changes depending on softmax == True or not
            if not self.softmax:

                for idx in range(conv_len-1):    
                    phi = torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers_2[idx+1], dim=(1,2,3)).squeeze()     
                    phi += torch.sum( self.pools[idx+1](self.B_syn[idx](layers_2[idx+1])) * layers[idx+2], dim=(1,2,3)).squeeze()
                    phis.append(phi)

                phi = torch.sum( self.pools[conv_len-1](self.synapses[conv_len-1](layers[conv_len-1])) * layers_2[conv_len], dim=(1,2,3)).squeeze()
                phi += torch.sum( self.B_syn[conv_len-1](layers_2[conv_len].view(mbs,-1)) * layers[conv_len+1], dim=1).squeeze()            
                phis.append(phi)            

                for idx in range(conv_len+1, tot_len-1):
                    phi = torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers_2[idx+1], dim=1).squeeze()
                    phi += torch.sum( self.B_syn[idx](layers_2[idx+1].view(mbs,-1)) * layers[idx+2], dim=1).squeeze()             
                    phis.append(phi)

                phi = torch.sum( self.synapses[-1](layers[-2].view(mbs,-1)) * layers_2[-1], dim=1).squeeze()
                if beta!=0.0:
                    if criterion.__class__.__name__.find('MSE')!=-1:
                        y = F.one_hot(y, num_classes=self.nc)
                        L = 0.5*criterion(layers_2[-1].float(), y.float()).sum(dim=1).squeeze()   
                    else:
                        L = criterion(layers_2[-1].float(), y).squeeze()             
                    phi -= beta*L
                phis.append(phi)

            else:
                #the output layer used for the prediction is no longer part of the system ! Summing until len(self.synapses) - 1 only
                for idx in range(conv_len-1):    
                    phi = torch.sum( self.pools[idx](self.synapses[idx](layers[idx])) * layers_2[idx+1], dim=(1,2,3)).squeeze()     
                    phi += torch.sum( self.pools[idx+1](self.B_syn[idx](layers_2[idx+1])) * layers[idx+2], dim=(1,2,3)).squeeze()
                    phis.append(phi)
            
                phi = torch.sum( self.pools[conv_len-1](self.synapses[conv_len-1](layers[conv_len-1])) * layers_2[conv_len], dim=(1,2,3)).squeeze()
                if bck_len>=conv_len:
                    phi += torch.sum( self.B_syn[conv_len-1](layers_2[conv_len].view(mbs,-1)) * layers[conv_len+1], dim=1).squeeze() 
                    phis.append(phi)            

                    for idx in range(conv_len, tot_len-2):
                        phi = torch.sum( self.synapses[idx](layers[idx].view(mbs,-1)) * layers_2[idx+1], dim=1).squeeze()
                        phi += torch.sum( self.B_syn[idx](layers_2[idx+1].view(mbs,-1)) * layers[idx+2], dim=1).squeeze()             
                        phis.append(phi)
                     
                    phi = torch.sum( self.synapses[-2](layers[-2].view(mbs,-1)) * layers_2[-1], dim=1).squeeze()
                    if beta!=0.0:
                        L = criterion(self.synapses[-1](layers_2[-1].view(mbs,-1)).float(), y).squeeze()             
                        phi -= beta*L
                    phis.append(phi)                    

                #the prediction is made with softmax[last weights[penultimate layer]]
                elif beta!=0.0:
                    L = criterion(self.synapses[-1](layers_2[-1].view(mbs,-1)).float(), y).squeeze()             
                    phi -= beta*L                       
                    phis.append(phi)
                else:
                    phis.append(phi)
        return phis
    

    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
 
        not_mse = (criterion.__class__.__name__.find('MSE')==-1)
        mbs = x.size(0)
        device = x.device     
        
        if check_thm:
            for t in range(T):
                phis = self.Phi(x, y, neurons, beta, criterion)
                for idx in range(len(neurons)-1):
                    init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                    grad = torch.autograd.grad(phis[idx], neurons[idx], grad_outputs=init_grad, create_graph=True)

                    neurons[idx] = self.activation( grad[0] )
                    neurons[idx].retain_grad()
             
                init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                grad = torch.autograd.grad(phis[-1], neurons[-1], grad_outputs=init_grad, create_graph=True)
                if not_mse and not(self.softmax):
                    neurons[-1] = grad[0]
                else:
                    neurons[-1] = self.activation( grad[0] )

                neurons[-1].retain_grad()
            #neurons[-1] = self.synapses[-1](neurons[-1].view(mbs,-1))
        else:
             for t in range(T):
                phis = self.Phi(x, y, neurons, beta, criterion)
                for idx in range(len(neurons)-1):
                    init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=True)
                    grad = torch.autograd.grad(phis[idx], neurons[idx], grad_outputs=init_grad, create_graph=False)

                    neurons[idx] = self.activation( grad[0] )
                    neurons[idx].requires_grad = True
             
                init_grad = torch.tensor([1 for i in range(mbs)], dtype=torch.float, device=device, requires_grad=False)
                grad = torch.autograd.grad(phis[-1], neurons[-1], grad_outputs=init_grad, create_graph=False)
                if not_mse and not(self.softmax):
                    neurons[-1] = grad[0]
                else:
                    neurons[-1] = self.activation( grad[0] )

                neurons[-1].requires_grad = True

        return neurons
       

    def init_neurons(self, mbs, device):
        
        neurons = []
        append = neurons.append
        size = self.in_size
        for idx in range(len(self.channels)-1): 
            size = int( (size + 2*self.paddings[idx] - self.kernels[idx])/self.strides[idx] + 1 )  # size after conv
            if self.pools[idx].__class__.__name__.find('Pool')!=-1:
                size = int( (size - self.pools[idx].kernel_size)/self.pools[idx].stride + 1 )  # size after Pool
            append(torch.zeros((mbs, self.channels[idx+1], size, size), requires_grad=True, device=device))

        size = size * size * self.channels[-1]
        
        if not self.softmax:
            for idx in range(len(self.fc)):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))
        else:
            # we remove the output layer from the system
            for idx in range(len(self.fc) - 1):
                append(torch.zeros((mbs, self.fc[idx]), requires_grad=True, device=device))            
            
        return neurons

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False, neurons_3=None):
        
        beta_1, beta_2 = betas
        
        self.zero_grad()            # p.grad is zero
        if not(check_thm):
            if neurons_3 is None: # neurons_3 is not None only when doing thirdphase with old VF (same update False) 
                phis_1 = self.Phi(x, y, neurons_1, beta_1, criterion)   # phi will habe the form s_* W s_*
            else:
                phis_1 = self.Phi(x, y, neurons_1, beta_1, criterion, neurons_2=neurons_2) # phi will have the form s_* W s_beta
        else:
            phis_1 = self.Phi(x, y, neurons_1, beta_2, criterion)

        if self.same_update:
            phis_2 = self.Phi(x, y, neurons_2, beta_2, criterion)  # Phi = s_beta W s_beta
        else:
            if neurons_3 is None:
                phis_2 = self.Phi(x, y, neurons_1, beta_2, criterion, neurons_2=neurons_2) # phi = s_* W s_beta
            else:
                phis_2 = self.Phi(x, y, neurons_1, beta_2, criterion, neurons_2=neurons_3) # phi = s_* W s_-beta
                
        for idx in range(len(neurons_1)):
            phi_1 = phis_1[idx].mean()
            phi_2 = phis_2[idx].mean()
            delta_phi = ((phi_2 - phi_1)/(beta_1 - beta_2))        
            delta_phi.backward() # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1)
        if self.same_update:
            with torch.no_grad():
                for idx in range(len(self.B_syn)):
                    common_update = 0.5*(self.B_syn[idx].weight.grad.data + self.synapses[idx+1].weight.grad.data)
                    self.B_syn[idx].weight.grad.data.copy_(common_update)
                    self.synapses[idx+1].weight.grad.data.copy_(common_update)
       

         

def check_gdu(model, x, y, T1, T2, betas, criterion, alg='EP'):
    # This function returns EP gradients and BPTT gradients for one training iteration
    #  given some labelled data (x, y), time steps for both phases and the loss
    
    # Initialize dictionnaries that will contain BPTT gradients and EP updates
    BPTT, EP = {}, {}
    if alg=='CEP':
        prev_p = {}

    for name, p in model.named_parameters():
        BPTT[name], EP[name] = [], []
        if alg=='CEP':
            prev_p[name] = p

    neurons = model.init_neurons(x.size(0), x.device)
    for idx in range(len(neurons)):
        BPTT['neurons_'+str(idx)], EP['neurons_'+str(idx)] = [], []
    
    # We first compute BPTT gradients
    # First phase up to T1-T2
    beta_1, beta_2 = betas
    neurons = model(x, y, neurons, T1-T2, beta=beta_1, criterion=criterion)
    ref_neurons = copy(neurons)
    

    # Last steps of the first phase
    for K in range(T2+1):

        neurons = model(x, y, neurons, K, beta=beta_1, criterion=criterion) # Running K time step 

        # detach data and neurons from the graph
        x = x.detach()
        x.requires_grad = True
        leaf_neurons = []
        for idx in range(len(neurons)):
            neurons[idx] = neurons[idx].detach()
            neurons[idx].requires_grad = True
            leaf_neurons.append(neurons[idx])

        neurons = model(x, y, neurons, T2-K, beta=beta_1, criterion=criterion, check_thm=True) # T2-K time step
        
        # final loss
        if criterion.__class__.__name__.find('MSE')!=-1:
            loss = (1/(2.0*x.size(0)))*criterion(neurons[-1].float(), F.one_hot(y, num_classes=model.nc).float()).sum(dim=1).squeeze()
        else:
            if not model.softmax:
                loss = (1/(x.size(0)))*criterion(neurons[-1].float(), y).squeeze()
            else:
                loss = (1/(x.size(0)))*criterion(model.synapses[-1](neurons[-1].view(x.size(0),-1)).float(), y).squeeze()

        # setting gradients field to zero before backward
        neurons_zero_grad(leaf_neurons)
        model.zero_grad()

        # Backpropagation through time
        loss.backward(torch.tensor([1 for i in range(x.size(0))], dtype=torch.float, device=x.device, requires_grad=True))

        # Collecting BPTT gradients : for parameters they are partial sums over T2-K time steps
        if K!=T2:
            for name, p in model.named_parameters():
                update = torch.empty_like(p).copy_(grad_or_zero(p))
                BPTT[name].append( update.unsqueeze(0) )  # unsqueeze for time dimension
                neurons = copy(ref_neurons) # Resetting the neurons to T1-T2 step
        if K!=0:
            for idx in range(len(leaf_neurons)):
                update = torch.empty_like(leaf_neurons[idx]).copy_(grad_or_zero(leaf_neurons[idx]))
                BPTT['neurons_'+str(idx)].append( update.mul(-x.size(0)).unsqueeze(0) )  # unsqueeze for time dimension

                                
    # Differentiating partial sums to get elementary parameter gradients
    for name, p in model.named_parameters():
        for idx in range(len(BPTT[name]) - 1):
            BPTT[name][idx] = BPTT[name][idx] - BPTT[name][idx+1]
        
    # Reverse the time
    for key in BPTT.keys():
        BPTT[key].reverse()
            
    # Now we compute EP gradients forward in time
    # Second phase done step by step
    for t in range(T2):
        neurons_pre = copy(neurons)                                          # neurons at time step t
        neurons = model(x, y, neurons, 1, beta=beta_2, criterion=criterion)  # neurons at time step t+1
        
        model.compute_syn_grads(x, y, neurons_pre, neurons, betas, criterion, check_thm=True)  # compute the EP parameter update

        if alg=='CEP':
            for p in model.parameters():
                p.data.add_(-1e-5 * p.grad.data)
        
        # Collect the EP updates forward in time
        for n, p in model.named_parameters():
            update = torch.empty_like(p).copy_(grad_or_zero(p))
            EP[n].append( update.unsqueeze(0) )                    # unsqueeze for time dimension
        for idx in range(len(neurons)):
            update = (neurons[idx] - neurons_pre[idx])/(beta_2 - beta_1)
            EP['neurons_'+str(idx)].append( update.unsqueeze(0) )  # unsqueeze for time dimension
        
    # Concatenating with respect to time dimension
    for key in BPTT.keys():
        BPTT[key] = torch.cat(BPTT[key], dim=0).detach()
        EP[key] = torch.cat(EP[key], dim=0).detach()
    
    if alg=='CEP':
        for name, p in model.named_parameters():
            p.data.copy_(prev_p[name])    

    return BPTT, EP
    


def RMSE(BPTT, EP):
    # print the root mean square error, and sign error between EP and BPTT gradients
    print('\nGDU check :')
    for key in BPTT.keys():
        K = BPTT[key].size(0)
        f_g = (EP[key] - BPTT[key]).pow(2).sum(dim=0).div(K).pow(0.5)
        f =  EP[key].pow(2).sum(dim=0).div(K).pow(0.5)
        g = BPTT[key].pow(2).sum(dim=0).div(K).pow(0.5)
        comp = f_g/(1e-10+torch.max(f,g))
        sign = torch.where(EP[key]*BPTT[key] < 0, torch.ones_like(EP[key]), torch.zeros_like(EP[key]))
        print(key.replace('.','_'), '\t RMSE =', round(comp.mean().item(), 4), '\t SIGN err =', round(sign.mean().item(), 4))
    print('\n')


def debug(model, prev_p, optimizer):
    optimizer.zero_grad()
    for (n, p) in model.named_parameters():
        idx = int(n[9]) 
        p.grad.data.copy_((prev_p[n] - p.data)/(optimizer.param_groups[idx]['lr']))
        p.data.copy_(prev_p[n])
    for i in range(len(model.synapses)):
        optimizer.param_groups[i]['lr'] = prev_p['lrs'+str(i)]
        #optimizer.param_groups[i]['weight_decay'] = prev_p['wds'+str(i)]
    optimizer.step()

        
def train(model, optimizer, train_loader, test_loader, T1, T2, betas, device, epochs, criterion, alg='EP', 
          random_sign=False, save=False, check_thm=False, path='', checkpoint=None, thirdphase = False, scheduler=None, cep_debug=False):
    
    mbs = train_loader.batch_size
    start = time.time()
    iter_per_epochs = math.ceil(len(train_loader.dataset)/mbs)
    beta_1, beta_2 = betas

    if checkpoint is None:
        train_acc = [10.0]
        test_acc = [10.0]
        best = 0.0
        epoch_sofar = 0
        angles = [90.0]
    else:
        train_acc = checkpoint['train_acc']
        test_acc = checkpoint['test_acc']    
        best = checkpoint['best']
        epoch_sofar = checkpoint['epoch']
        angles = checkpoint['angles'] if 'angles' in checkpoint.keys() else []

    for epoch in range(epochs):
        run_correct = 0
        run_total = 0
        model.train()

        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            #if alg=='CEP' and cep_debug:
            #    x = x.double()   
    
            neurons = model.init_neurons(x.size(0), device)
            if alg=='EP' or alg=='CEP':
                # First phase
                neurons = model(x, y, neurons, T1, beta=beta_1, criterion=criterion)
                neurons_1 = copy(neurons)
            elif alg=='BPTT':
                neurons = model(x, y, neurons, T1-T2, beta=0.0, criterion=criterion)           
                # detach data and neurons from the graph
                x = x.detach()
                x.requires_grad = True
                for k in range(len(neurons)):
                    neurons[k] = neurons[k].detach()
                    neurons[k].requires_grad = True

                neurons = model(x, y, neurons, T2, beta=0.0, criterion=criterion, check_thm=True) # T2 time step

            # Predictions for running accuracy
            with torch.no_grad():
                if not model.softmax:
                    pred = torch.argmax(neurons[-1], dim=1).squeeze()
                else:
                    #WATCH OUT: prediction is different when softmax == True
                    pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1), dim = 1).squeeze()

                run_correct += (y == pred).sum().item()
                run_total += x.size(0)
                if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)) and save:
                    plot_neural_activity(neurons, path)
            
            if alg=='EP':
                # Second phase
                if random_sign and (beta_1==0.0):
                    rnd_sgn = 2*np.random.randint(2) - 1
                    betas = beta_1, rnd_sgn*beta_2
                    beta_1, beta_2 = betas
            
                neurons = model(x, y, neurons, T2, beta = beta_2, criterion=criterion)
                neurons_2 = copy(neurons)
                # Third phase (if we approximate f' as f'(x) = (f(x+h) - f(x-h))/2h)
                if thirdphase:
                    #come back to the first equilibrium
                    neurons = copy(neurons_1)
                    neurons = model(x, y, neurons, T2, beta = - beta_2, criterion=criterion)
                    neurons_3 = copy(neurons)
                    if not(isinstance(model, VF_CNN)):
                        model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, - beta_2), criterion)
                    else:
                        if model.same_update:
                            model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, - beta_2), criterion)
                        else:    
                            model.compute_syn_grads(x, y, neurons_1, neurons_2, (beta_2, - beta_2), criterion, neurons_3=neurons_3)
                else:
                    model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)

                optimizer.step()      

            elif alg=='CEP':
                if random_sign and (beta_1==0.0):
                    rnd_sgn = 2*np.random.randint(2) - 1
                    betas = beta_1, rnd_sgn*beta_2
                    beta_1, beta_2 = betas

                # second phase
                if cep_debug:
                    prev_p = {}
                    for (n, p) in model.named_parameters():
                        prev_p[n] = p.clone().detach()
                    for i in range(len(model.synapses)):
                        prev_p['lrs'+str(i)] = optimizer.param_groups[i]['lr']
                        prev_p['wds'+str(i)] = optimizer.param_groups[i]['weight_decay']
                        optimizer.param_groups[i]['lr'] *= 6e-5
                        #optimizer.param_groups[i]['weight_decay'] = 0.0
                                        
                for k in range(T2):
                    neurons = model(x, y, neurons, 1, beta = beta_2, criterion=criterion)   # one step
                    neurons_2  = copy(neurons)
                    model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)   # compute cep update between 2 consecutive steps 
                    for (n, p) in model.named_parameters():
                        p.grad.data.div_( (1 - optimizer.param_groups[int(n[9])]['lr']*optimizer.param_groups[int(n[9])]['weight_decay'])**(T2-1-k)  ) 
                    optimizer.step()                                                        # update weights 
                    neurons_1 = copy(neurons)
               
                if cep_debug:
                    debug(model, prev_p, optimizer)
 
                if thirdphase:    
                    neurons = model(x, y, neurons, T2, beta = 0.0, criterion=criterion)     # come back to s*
                    neurons_2 = copy(neurons)
                    for k in range(T2):
                        neurons = model(x, y, neurons, 1, beta = -beta_2, criterion=criterion)
                        neurons_3 = copy(neurons)
                        model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, -beta_2), criterion)
                        optimizer.step()
                        neurons_2 = copy(neurons)

            elif alg=='BPTT':
         
                # final loss
                if criterion.__class__.__name__.find('MSE')!=-1:
                    loss = 0.5*criterion(neurons[-1].float(), F.one_hot(y, num_classes=model.nc).float()).sum(dim=1).mean().squeeze()
                else:
                    if not model.softmax:
                        loss = criterion(neurons[-1].float(), y).mean().squeeze()
                    else:
                        loss = criterion(model.synapses[-1](neurons[-1].view(x.size(0),-1)).float(), y).mean().squeeze()
                # setting gradients field to zero before backward
                model.zero_grad()

                # Backpropagation through time
                loss.backward()
                optimizer.step()
    
            if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)):
                run_acc = run_correct/run_total
                print('Epoch :', round(epoch_sofar+epoch+(idx/iter_per_epochs), 2),
                      '\tRun train acc :', round(run_acc,3),'\t('+str(run_correct)+'/'+str(run_total)+')\t',
                      timeSince(start, ((idx+1)+epoch*iter_per_epochs)/(epochs*iter_per_epochs)))
                if isinstance(model, VF_CNN): 
                    angle = model.angle()
                    print('angles ',angle)
                if check_thm and alg!='BPTT':
                    BPTT, EP = check_gdu(model, x[0:5,:], y[0:5], T1, T2, betas, criterion, alg=alg)
                    RMSE(BPTT, EP)
    
        if scheduler is not None: # learning rate decay step
            if epoch+epoch_sofar < scheduler.T_max:
                scheduler.step()

        test_correct = evaluate(model, test_loader, T1, device)
        test_acc_t = test_correct/(len(test_loader.dataset))
        if save:
            test_acc.append(100*test_acc_t)
            train_acc.append(100*run_acc)
            if isinstance(model, VF_CNN):
                angle = model.angle()
                angles.append(angle)
            if test_correct > best:
                best = test_correct
                save_dic = {'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                            'train_acc': train_acc, 'test_acc': test_acc, 
                            'best': best, 'epoch': epoch_sofar+epoch+1}
                save_dic['angles'] = angles
                save_dic['scheduler'] = scheduler.state_dict() if scheduler is not None else None
                torch.save(save_dic,  path + '/checkpoint.tar')
                torch.save(model, path + '/model.pt')
            plot_acc(train_acc, test_acc, path)        
    
    if save:
        save_dic = {'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                    'train_acc': train_acc, 'test_acc': test_acc, 
                    'best': best, 'epoch': epochs}
        save_dic['angles'] = angles
        save_dic['scheduler'] = scheduler.state_dict() if scheduler is not None else None
        torch.save(save_dic,  path + '/final_checkpoint.tar')
        torch.save(model, path + '/final_model.pt')
 
def adv_train(model, optimizer, train_loader, test_loader, T1, T2, betas, device, epochs, criterion, epsilon, train_attack_step, alg='EP', 
          random_sign=False, save=False, check_thm=False, path='', checkpoint=None, thirdphase = False, scheduler=None, cep_debug=False):
    global attack_param
    attack_param = 0
    mean = np.array([0.4914, 0.4822, 0.4465])[None,:,None,None]
    std=np.array([3*0.2023, 3*0.1994, 3*0.2010])[None,:,None,None]
    last_layer = 1
    if criterion.__class__.__name__.find('MSE')!=-1:
            art_model = PyTorchClassifier(model, torch.nn.MSELoss(), (3, 32, 32), 10, T1, T2, clip_values=None,preprocessing=(mean,std))
    else:
        art_model = PyTorchClassifier(model, torch.nn.CrossEntropyLoss(), (3, 32, 32), 10, T1=100, T2=train_attack_step, last_layer=last_layer, clip_values=None,preprocessing=(mean,std))
    attack = ProjectedGradientDescentPyTorch(art_model, 2, epsilon, 2.5 * epsilon / 20, max_iter=20, batch_size=train_loader.batch_size, verbose=False) 
    mbs = train_loader.batch_size
    start = time.time()
    iter_per_epochs = math.ceil(len(train_loader.dataset)/mbs)
    beta_1, beta_2 = betas

    if checkpoint is None:
        train_acc = [10.0]
        test_acc = [10.0]
        best = 0.0
        epoch_sofar = 0
        angles = [90.0]
    else:
        train_acc = checkpoint['train_acc']
        test_acc = checkpoint['test_acc']    
        best = checkpoint['best']
        epoch_sofar = checkpoint['epoch']
        angles = checkpoint['angles'] if 'angles' in checkpoint.keys() else []

    for epoch in range(epochs):
        run_correct = 0
        run_total = 0
        model.train()

        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            #if alg=='CEP' and cep_debug:
            #    x = x.double()   
            attack_param = 1
            x = torch.from_numpy(attack.generate(x.cpu().numpy())).to(device)
            attack_param = 0
            neurons = model.init_neurons(x.size(0), device)
            if alg=='EP' or alg=='CEP':
                # First phase
                neurons = model(x, y, neurons, T1, beta=beta_1, criterion=criterion)
                neurons_1 = copy(neurons)
            elif alg=='BPTT':
                neurons = model(x, y, neurons, T1-T2, beta=0.0, criterion=criterion)           
                # detach data and neurons from the graph
                x = x.detach()
                x.requires_grad = True
                for k in range(len(neurons)):
                    neurons[k] = neurons[k].detach()
                    neurons[k].requires_grad = True

                neurons = model(x, y, neurons, T2, beta=0.0, criterion=criterion, check_thm=True) # T2 time step

            # Predictions for running accuracy
            with torch.no_grad():
                if not model.softmax:
                    pred = torch.argmax(neurons[-1], dim=1).squeeze()
                else:
                    #WATCH OUT: prediction is different when softmax == True
                    pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1), dim = 1).squeeze()

                run_correct += (y == pred).sum().item()
                run_total += x.size(0)
                if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)) and save:
                    plot_neural_activity(neurons, path)
            
            if alg=='EP':
                # Second phase
                if random_sign and (beta_1==0.0):
                    rnd_sgn = 2*np.random.randint(2) - 1
                    betas = beta_1, rnd_sgn*beta_2
                    beta_1, beta_2 = betas
            
                neurons = model(x, y, neurons, T2, beta = beta_2, criterion=criterion)
                neurons_2 = copy(neurons)
                # Third phase (if we approximate f' as f'(x) = (f(x+h) - f(x-h))/2h)
                if thirdphase:
                    #come back to the first equilibrium
                    neurons = copy(neurons_1)
                    neurons = model(x, y, neurons, T2, beta = - beta_2, criterion=criterion)
                    neurons_3 = copy(neurons)
                    if not(isinstance(model, VF_CNN)):
                        model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, - beta_2), criterion)
                    else:
                        if model.same_update:
                            model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, - beta_2), criterion)
                        else:    
                            model.compute_syn_grads(x, y, neurons_1, neurons_2, (beta_2, - beta_2), criterion, neurons_3=neurons_3)
                else:
                    model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)

                optimizer.step()      

            elif alg=='CEP':
                if random_sign and (beta_1==0.0):
                    rnd_sgn = 2*np.random.randint(2) - 1
                    betas = beta_1, rnd_sgn*beta_2
                    beta_1, beta_2 = betas

                # second phase
                if cep_debug:
                    prev_p = {}
                    for (n, p) in model.named_parameters():
                        prev_p[n] = p.clone().detach()
                    for i in range(len(model.synapses)):
                        prev_p['lrs'+str(i)] = optimizer.param_groups[i]['lr']
                        prev_p['wds'+str(i)] = optimizer.param_groups[i]['weight_decay']
                        optimizer.param_groups[i]['lr'] *= 6e-5
                        #optimizer.param_groups[i]['weight_decay'] = 0.0
                                        
                for k in range(T2):
                    neurons = model(x, y, neurons, 1, beta = beta_2, criterion=criterion)   # one step
                    neurons_2  = copy(neurons)
                    model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)   # compute cep update between 2 consecutive steps 
                    for (n, p) in model.named_parameters():
                        p.grad.data.div_( (1 - optimizer.param_groups[int(n[9])]['lr']*optimizer.param_groups[int(n[9])]['weight_decay'])**(T2-1-k)  ) 
                    optimizer.step()                                                        # update weights 
                    neurons_1 = copy(neurons)
               
                if cep_debug:
                    debug(model, prev_p, optimizer)
 
                if thirdphase:    
                    neurons = model(x, y, neurons, T2, beta = 0.0, criterion=criterion)     # come back to s*
                    neurons_2 = copy(neurons)
                    for k in range(T2):
                        neurons = model(x, y, neurons, 1, beta = -beta_2, criterion=criterion)
                        neurons_3 = copy(neurons)
                        model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, -beta_2), criterion)
                        optimizer.step()
                        neurons_2 = copy(neurons)

            elif alg=='BPTT':
         
                # final loss
                if criterion.__class__.__name__.find('MSE')!=-1:
                    loss = 0.5*criterion(neurons[-1].float(), F.one_hot(y, num_classes=model.nc).float()).sum(dim=1).mean().squeeze()
                else:
                    if not model.softmax:
                        loss = criterion(neurons[-1].float(), y).mean().squeeze()
                    else:
                        loss = criterion(model.synapses[-1](neurons[-1].view(x.size(0),-1)).float(), y).mean().squeeze()
                # setting gradients field to zero before backward
                model.zero_grad()

                # Backpropagation through time
                loss.backward()
                optimizer.step()
    
            if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)):
                run_acc = run_correct/run_total
                print('Epoch :', round(epoch_sofar+epoch+(idx/iter_per_epochs), 2),
                      '\tRun train acc :', round(run_acc,3),'\t('+str(run_correct)+'/'+str(run_total)+')\t',
                      timeSince(start, ((idx+1)+epoch*iter_per_epochs)/(epochs*iter_per_epochs)))
                if isinstance(model, VF_CNN): 
                    angle = model.angle()
                    print('angles ',angle)
                if check_thm and alg!='BPTT':
                    BPTT, EP = check_gdu(model, x[0:5,:], y[0:5], T1, T2, betas, criterion, alg=alg)
                    RMSE(BPTT, EP)
    
        if scheduler is not None: # learning rate decay step
            if epoch+epoch_sofar < scheduler.T_max:
                scheduler.step()

        test_correct = evaluate(model, test_loader, T1, device)
        test_acc_t = test_correct/(len(test_loader.dataset))
        if save:
            test_acc.append(100*test_acc_t)
            train_acc.append(100*run_acc)
            if isinstance(model, VF_CNN):
                angle = model.angle()
                angles.append(angle)
            if test_correct > best:
                best = test_correct
                save_dic = {'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                            'train_acc': train_acc, 'test_acc': test_acc, 
                            'best': best, 'epoch': epoch_sofar+epoch+1}
                save_dic['angles'] = angles
                save_dic['scheduler'] = scheduler.state_dict() if scheduler is not None else None
                torch.save(save_dic,  path + '/checkpoint_Train_Eps_%d.tar'%(int(epsilon*1000)))
                torch.save(model, path + '/model_Train_Eps_%d.pt'%(int(epsilon*1000)))
            plot_acc(train_acc, test_acc, path)        
    
    if save:
        save_dic = {'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                    'train_acc': train_acc, 'test_acc': test_acc, 
                    'best': best, 'epoch': epochs}
        save_dic['angles'] = angles
        save_dic['scheduler'] = scheduler.state_dict() if scheduler is not None else None
        torch.save(save_dic,  path + '/final_checkpoint.tar')
        torch.save(model, path + '/final_model.pt')

def evaluate(model, loader, T, device):
    # Evaluate the model on a dataloader with T steps for the dynamics
    model.eval()
    correct=0
    phase = 'Train' if loader.dataset.train else 'Test'
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        neurons = model.init_neurons(x.size(0), device)
        neurons = model(x, y, neurons, T) # dynamics for T time steps

        if not model.softmax:
            pred = torch.argmax(neurons[-1], dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
        else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
            pred = torch.argmax(F.softmax(model.synapses[-1](neurons[-1].view(x.size(0),-1)), dim = 1), dim = 1).squeeze()

        correct += (y == pred).sum().item()

    acc = correct/len(loader.dataset) 
    print(phase+' accuracy :\t', acc)   
    return correct
def characteristics(model, loader, T, device, load_path):
    # Evaluate the model on a dataloader with T steps for the dynamics
    model.eval()
    global characteristic_param, characteristic_time
    #characteristic_time = np.zeros((T,len(loader.dataset),10))
    characteristic_param = 1
    characteristic_time_2 = []
    characteristic_time_3 = []
    saturation_time = np.zeros((250,2))
    predictions = []
    correct=0
    phase = 'Train' if loader.dataset.train else 'Test'
    for idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        neurons = model.init_neurons(x.size(0), device)
        neurons = model(x, y, neurons, T) # dynamics for T time steps
        characteristic_time_2 = []
        for j in range(250):
            temp = characteristic_time[250*idx+j].detach().cpu().numpy()
            characteristic_time_2.append(temp)
        pred = np.argmax(characteristic_time_2,axis=2)
        y_ones = np.repeat(np.asarray([y.cpu().numpy()]),repeats=250,axis=0)
        saturation_time[:,1] += np.sum(y_ones==pred,axis=1)
        
        if idx==0:
            characteristic_time_3 = np.transpose(characteristic_time_2,(1,0,2))
            predictions = (y_ones==pred).T
        else:
            characteristic_time_3 = np.vstack((characteristic_time_3,np.transpose(characteristic_time_2,(1,0,2))))
            predictions = np.vstack((predictions,(y_ones==pred).T))
    predictions =  np.asarray(predictions)
    print(characteristic_time_3.shape,predictions.shape)
    saturation_time[:,1] = saturation_time[:,1]/len(loader.dataset)
    saturation_time[:,0] = np.arange(250)+1
    np.save(load_path+'Characteristic_Logits.npy',characteristic_time_3)
    np.savetxt(load_path+'Characteristic_Time.txt',saturation_time) 
    np.savetxt(load_path+'Characteristic_Predictions.txt',predictions) 
    return correct
def attack(model, loader, T, criterion, device,load_path,arg_softmax, image_print, attack_step, predict_step,attack_norm):
    # Evaluate the model on a dataloader with T steps for the dynamics
    global attack_param
    model.eval()
    correct = 0
    start = time.time()
    phase = 'Attack Train' if loader.dataset.train else 'Attack Test'
    #eps = np.linspace(1,10,20)
    if(attack_norm==100):
        eps = np.logspace(np.log10(0.0001),np.log10(0.1),10)   
        eps = np.insert(eps,0,0.031)
        eps = np.sort(eps)
    else:
        eps = np.logspace(np.log10(0.05),np.log10(3),10)
        eps = np.insert(eps,0,1)
        eps = np.sort(eps)
        
    print(eps)
    
    
    accuracy_values = np.zeros((len(eps),3))
    mean = np.array([0.4914, 0.4822, 0.4465])[None,:,None,None]
    std=np.array([3*0.2023, 3*0.1994, 3*0.2010])[None,:,None,None]
    predict_timesteps = predict_step
    attack_timesteps = attack_step
    last_layer = 1
    print(arg_softmax)
    if(attack_norm==100):
        attack_norm_str = 100
        attack_norm = np.inf
    else:
        attack_norm_str = attack_norm
    for i in range(len(eps)):
        neurons = model.init_neurons(loader.batch_size, device)
        print(loader.batch_size)
        if criterion.__class__.__name__.find('MSE')!=-1:
            art_model = PyTorchClassifier(model, torch.nn.MSELoss(), (3, 32, 32), 10, predict_timesteps, attack_timesteps, last_layer=last_layer, clip_values=None,preprocessing=(mean,std))
        else:
            art_model = PyTorchClassifier(model, torch.nn.CrossEntropyLoss(), (3, 32, 32), 10, predict_timesteps, attack_timesteps, last_layer=last_layer, clip_values=None,preprocessing=(mean,std))
        attack = ProjectedGradientDescentPyTorch(art_model, attack_norm, eps[i], 2.5 * eps[i] / 20, max_iter=20, batch_size=loader.batch_size, verbose=False)   
        correct = 0
        correct_adv = 0 
        if(image_print==0):
            mbs = loader.batch_size
            run_total = 0
            iter_per_epochs = math.ceil(len(loader.dataset)/mbs)
            cifar_adv, labels = [], []
            confidence = []
            for idx, (x, y) in enumerate(loader):
                attack_param = 1          
                x, y = x.to(device), y.to(device) 
                neurons = 0
                neurons = torch.from_numpy(art_model.predict(x.cpu().numpy())).to(device)
                if not arg_softmax:
                    pred = torch.argmax(neurons, dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
                else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
                    if(last_layer==0):
                        pred = torch.argmax(F.softmax(model.synapses[-1](neurons.view(x.size(0),-1)), dim = 1), dim = 1).squeeze()
                    else:
                        pred = torch.argmax(neurons, dim = 1).squeeze()
                correct += (y == pred).sum().item()   
                
                x_adv = attack.generate(x.cpu().numpy())
                model.zero_grad()
                neurons_adv = torch.from_numpy(art_model.predict(x_adv)).to(device)
                
                if not arg_softmax:
                    pred = torch.argmax(neurons_adv, dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
                else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
                    if(last_layer==0):
                        pred = torch.argmax(F.softmax(model.synapses[-1](neurons_adv.view(x.size(0),-1)), dim = 1), dim = 1).squeeze()
                        confidence.append(model.synapses[-1](neurons_adv.view(x.size(0),-1)).cpu().detach().numpy())
                    else:
                        pred = torch.argmax(neurons_adv, dim = 1).squeeze()
                        confidence.append(neurons_adv.cpu().numpy())
                cifar_adv.append(x_adv)
                labels.append(pred.cpu().numpy())    
                correct_adv += (y == pred).sum().item()
                model.zero_grad()
                run_total += x.size(0)
                
                run_acc = correct/run_total
                adv_run_acc = correct_adv/run_total
                print(
                f'Epsilon: {eps[i]}; ',
                f'Train Acc: {round(run_acc, 2)}; ',
                f'Adversarial Acc: {round(adv_run_acc, 2)}; ',
                f'Time Elapsed: {time.time()-start}'
                )
                
            cifar_adv = np.reshape(np.asarray(cifar_adv),(-1,3,32,32))
            labels = np.asarray(labels).flatten()
            confidence = np.reshape(np.asarray(confidence),(-1,10))
            if not last_layer:
                np.save(load_path + 'White_Norm_%d_Adversarial_Images_Eps_%d_Attack_T_%d_Predict_T_%d.npy'%(attack_norm_str, int(eps[i]*1000),attack_timesteps,predict_timesteps),
                    cifar_adv)
                np.save(load_path + 'White_Norm_%d_Labels_Eps_%d_Attack_T_%d_Predict_T_%d.npy'%(attack_norm_str, int(eps[i]*1000),attack_timesteps,predict_timesteps),
                    labels)
                np.save(load_path + 'White_Norm_%d_Confidence_Eps_%d_Attack_T_%d_Predict_T_%d.npy'%(attack_norm_str, int(eps[i]*1000),attack_timesteps,predict_timesteps),
                    confidence)
            else: 
                np.save(load_path + 'White_Last_Layer_Norm_%d_Adversarial_Images_Eps_%d_Attack_T_%d_Predict_T_%d.npy'%(attack_norm_str,int(eps[i]*1000),attack_timesteps,predict_timesteps),
                    cifar_adv)
                np.save(load_path + 'White_Last_Layer_Norm_%d_Labels_Eps_%d_Attack_T_%d_Predict_T_%d.npy'%(attack_norm_str, int(eps[i]*1000),attack_timesteps,predict_timesteps),
                    labels)
                np.save(load_path + 'White_Last_Layer_Norm_%d_Confidence_Eps_%d_Attack_T_%d_Predict_T_%d.npy'%(attack_norm_str, int(eps[i]*1000),attack_timesteps,predict_timesteps),
                    confidence)
            acc = correct/run_total
            acc_adv = correct_adv/run_total
            print(
            f'Epsilon: {eps[i]}; ',
            f'Train Acc: {round(acc, 2)}; ',
            f'Adversarial Acc: {round(acc_adv, 2)}; ',
            f'Time Elapsed: {time.time()-start}'
            )
            accuracy_values[i] = eps[i],acc,acc_adv
            if not last_layer:
                np.savetxt(load_path+'/White_Norm_%d_Adversarial_Accuracy_Attack_T_%d_Predict_T_%d.txt'%(attack_norm_str,attack_timesteps,predict_timesteps),accuracy_values,fmt='%.6f')
            else:
                np.savetxt(load_path+'/White_Last_Layer_Norm_%d_Adversarial_Accuracy_Attack_T_%d_Predict_T_%d.txt'%(attack_norm_str,attack_timesteps,predict_timesteps),accuracy_values,fmt='%.6f')
            
        else:
            eps = np.array([0.5,1.0,2.0,3.0,7.0])
            x = torch.load('/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/Test_Image.pt')[None,:,:,:].to(device)
            neurons = torch.from_numpy(art_model.predict(x.cpu().numpy())).to(device)
            x_adv = attack.generate(x.cpu().numpy())
            neurons_adv = torch.from_numpy(art_model.predict(x_adv)).to(device)
            torch.save(torch.from_numpy(x_adv),load_path+'/Adv_Image_Eps_%d.pt'%(i))
            torch.save(neurons_adv,load_path+'/Adv_Label_Eps_%d.pt'%(i))
            if not arg_softmax:
                pred = torch.argmax(neurons_adv, dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
                print(neurons_adv.shape)
            else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
                pred = torch.argmax(F.softmax(model.synapses[-1](neurons_adv.view(x.size(0),-1)), dim = 1), dim = 1).squeeze()
                print(neurons_adv.view(x.size(0),-1).shape,neurons_adv.shape,model.synapses[-1])
            pred =pred.cpu().unsqueeze(0).numpy()
            
            np.savetxt(load_path+'/Label_Eps_%d.txt'%i,pred,fmt='%d')
            
               
    
    return correct
def img_print(model, loader, T, criterion, device,load_path,arg_softmax, image_print, attack_step, predict_step,attack_norm):
    # Evaluate the model on a dataloader with T steps for the dynamics
    global attack_param
    model.eval()
    
        
    cifar_adv, labels = [], []
    confidence = []
    for idx, (x, y) in enumerate(loader):        
        cifar_adv.append(x.numpy())
        labels.append(y.numpy())    
    cifar_adv = np.reshape(np.asarray(cifar_adv),(-1,3,32,32))
    labels = np.asarray(labels).flatten()
    load_path = '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/Natural_Images/'
    np.save(load_path + 'Images.npy', cifar_adv)
    np.save(load_path + 'Labels.npy', labels)
    
    
    return 0
def attack_resume(model, loader, T, criterion, device,load_path,arg_softmax, image_print, attack_step, predict_step,attack_norm):
    # Evaluate the model on a dataloader with T steps for the dynamics
    global attack_param
    model.eval()
    correct = 0
    start = time.time()
    phase = 'Attack Train' if loader.dataset.train else 'Attack Test'
    #eps = np.linspace(1,10,20)
    if(attack_norm==100):
        eps = np.logspace(np.log10(0.0001),np.log10(0.1),10)       
    else:
        eps = np.logspace(np.log10(0.05),np.log10(3),10)
        eps = np.insert(eps,0,1)
        eps = np.sort(eps)
    #eps = [1]
    eps = eps[:5]
    print(eps)
    predict_timesteps = predict_step
    attack_timesteps = attack_step
    
    accuracy_values = np.loadtxt(load_path+'/White_Last_Layer_Adversarial_Accuracy_Attack_T_%d_Predict_T_%d.txt'%(attack_timesteps,predict_timesteps))
    mean = np.array([0.4914, 0.4822, 0.4465])[None,:,None,None]
    std=np.array([3*0.2023, 3*0.1994, 3*0.2010])[None,:,None,None]
    
    last_layer = 1
    attack_norm = 2
    for i in range(len(eps)):
        
        neurons = model.init_neurons(loader.batch_size, device)
        print(loader.batch_size)
        if criterion.__class__.__name__.find('MSE')!=-1:
            art_model = PyTorchClassifier(model, torch.nn.MSELoss(), (3, 32, 32), 10, predict_timesteps, attack_timesteps, clip_values=None,preprocessing=(mean,std))
        else:
            art_model = PyTorchClassifier(model, torch.nn.CrossEntropyLoss(), (3, 32, 32), 10, predict_timesteps, attack_timesteps, last_layer=last_layer, clip_values=None,preprocessing=(mean,std))
        attack = ProjectedGradientDescentPyTorch(art_model, attack_norm, eps[i], 2.5 * eps[i] / 20, max_iter=20, batch_size=loader.batch_size, verbose=False)   
        correct = 0
        correct_adv = 0 
        if(image_print==0):
            mbs = loader.batch_size
            run_total = 0
            iter_per_epochs = math.ceil(len(loader.dataset)/mbs)
            cifar_adv, labels = [], []
            confidence = []
            for idx, (x, y) in enumerate(loader):
                attack_param = 1          
                x, y = x.to(device), y.to(device) 
                neurons = 0
                neurons = torch.from_numpy(art_model.predict(x.cpu().numpy())).to(device)
                if not arg_softmax:
                    pred = torch.argmax(neurons, dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
                else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
                    if(last_layer==0):
                        pred = torch.argmax(F.softmax(model.synapses[-1](neurons.view(x.size(0),-1)), dim = 1), dim = 1).squeeze()
                    else:
                        pred = torch.argmax(neurons, dim = 1).squeeze()
                correct += (y == pred).sum().item()   
                
                x_adv = attack.generate(x.cpu().numpy())
                model.zero_grad()
                neurons_adv = torch.from_numpy(art_model.predict(x_adv)).to(device)
                
                if not arg_softmax:
                    pred = torch.argmax(neurons_adv, dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
                else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
                    if(last_layer==0):
                        pred = torch.argmax(F.softmax(model.synapses[-1](neurons_adv.view(x.size(0),-1)), dim = 1), dim = 1).squeeze()
                        confidence.append(model.synapses[-1](neurons_adv.view(x.size(0),-1)).cpu().detach().numpy())
                    else:
                        pred = torch.argmax(neurons_adv, dim = 1).squeeze()
                        confidence.append(neurons_adv.cpu().numpy())
                cifar_adv.append(x_adv)
                labels.append(pred.cpu().numpy())    
                correct_adv += (y == pred).sum().item()
                model.zero_grad()
                run_total += x.size(0)
                
                run_acc = correct/run_total
                adv_run_acc = correct_adv/run_total
                if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)):
                    print(
                    f'Epsilon: {eps[i]}; ',
                    f'Train Acc: {round(run_acc, 2)}; ',
                    f'Adversarial Acc: {round(adv_run_acc, 2)}; ',
                    f'Time Elapsed: {time.time()-start}'
                    )
                
            cifar_adv = np.reshape(np.asarray(cifar_adv),(-1,3,32,32))
            labels = np.asarray(labels).flatten()
            confidence = np.reshape(np.asarray(confidence),(-1,10))
            if not last_layer:
                np.save(load_path + 'White_Adversarial_Images_Eps_%d_Attack_T_%d_Predict_T_%d.npy'%( int(eps[i]*1000),attack_timesteps,predict_timesteps),
                    cifar_adv)
                np.save(load_path + 'White_Labels_Eps_%d_Attack_T_%d_Predict_T_%d.npy'%( int(eps[i]*1000),attack_timesteps,predict_timesteps),
                    labels)
                np.save(load_path + 'White_Confidence_Eps_%d_Attack_T_%d_Predict_T_%d.npy'%( int(eps[i]*1000),attack_timesteps,predict_timesteps),
                    confidence)
            else: 
                np.save(load_path + 'White_Last_Layer_Adversarial_Images_Eps_%d_Attack_T_%d_Predict_T_%d.npy'%(int(eps[i]*1000),attack_timesteps,predict_timesteps),
                    cifar_adv)
                np.save(load_path + 'White_Last_Layer_Labels_Eps_%d_Attack_T_%d_Predict_T_%d.npy'%( int(eps[i]*1000),attack_timesteps,predict_timesteps),
                    labels)
                np.save(load_path + 'White_Last_Layer_Confidence_Eps_%d_Attack_T_%d_Predict_T_%d.npy'%( int(eps[i]*1000),attack_timesteps,predict_timesteps),
                    confidence)
            acc = correct/run_total
            acc_adv = correct_adv/run_total
            print(
            f'Epsilon: {eps[i]}; ',
            f'Train Acc: {round(acc, 2)}; ',
            f'Adversarial Acc: {round(acc_adv, 2)}; ',
            f'Time Elapsed: {time.time()-start}'
            )
            accuracy_values[i] = eps[i],acc,acc_adv
            if not last_layer:
                np.savetxt(load_path+'/White_Adversarial_Accuracy_Attack_T_%d_Predict_T_%d.txt'%(attack_timesteps,predict_timesteps),accuracy_values,fmt='%.6f')
            else:
                np.savetxt(load_path+'/White_Last_Layer_Adversarial_Accuracy_Attack_T_%d_Predict_T_%d.txt'%(attack_timesteps,predict_timesteps),accuracy_values,fmt='%.6f')
            
    
            
               
    
    return correct
def hsj_attack(model, loader, T, criterion, device,load_path,arg_softmax, image_print, attack_step, predict_step,attack_norm):
    # Evaluate the model on a dataloader with T steps for the dynamics
    global attack_param
    model.eval()
    correct = 0
    start = time.time()
    phase = 'Attack Train' if loader.dataset.train else 'Attack Test'
    #eps = np.linspace(1,10,20)
    max_evals = [100,1000,10000]
    if(attack_norm==100):
        test_eps = np.logspace(np.log10(0.0001),np.log10(0.1),10)  
        #test_eps = np.insert(test_eps,0,1)
        test_eps = np.insert(test_eps,0,8/255)
        test_eps = np.sort(test_eps)
    else:

        test_eps = np.logspace(np.log10(0.05),np.log10(3),10)
        test_eps = np.insert(test_eps,0,1)
        test_eps = np.insert(test_eps,0,0.005)
        test_eps = np.sort(test_eps)
    norm_str = attack_norm
    if(attack_norm==100):
        attack_norm=np.inf
    print(attack_norm)
    global predict_counter
    predict_counter = 0
    def predict_square(x):
        global predict_counter 
        predict_counter +=1
        x_c = torch.from_numpy(x).to(device)
        neurons = model.init_neurons(x_c.size(0), device)
        neurons = model(x_c,T=T)
        y_pred = F.softmax(model.synapses[-1](neurons[-1].view(x_c.size(0),-1)), dim = 1)
        return y_pred.detach().cpu().numpy()
    
    accuracy_values = np.zeros((len(test_eps),3))
    mean = np.array([0.4914, 0.4822, 0.4465])[None,:,None,None]
    std=np.array([3*0.2023, 3*0.1994, 3*0.2010])[None,:,None,None]
    predict_timesteps = predict_step
    attack_timesteps = attack_step
    max_iter = 1
    
    for i in range(3,len(test_eps)):
        neurons = model.init_neurons(loader.batch_size, device)
        print(loader.batch_size)
        if criterion.__class__.__name__.find('MSE')!=-1:
            art_model = PyTorchClassifier(model, torch.nn.MSELoss(), (3, 32, 32), 10, predict_timesteps, attack_timesteps, clip_values=None,preprocessing=(mean,std))
        else:
            
            base_model = BlackBoxClassifierNeuralNetwork(predict_square, input_shape=(3, 32, 32), nb_classes=10,clip_values=(0,1),preprocessing=(mean,std))
        attack = SquareAttack(base_model, norm = attack_norm,nb_restarts=1, eps = test_eps[i],max_iter=max_evals[max_iter], batch_size=loader.batch_size, verbose=False)   
        correct = 0
        correct_adv = 0 
        if(image_print==0):
            mbs = loader.batch_size
            run_total = 0
            iter_per_epochs = math.ceil(len(loader.dataset)/mbs)
            cifar_adv, labels = [], []
            confidence = []
            for idx, (x, y) in enumerate(loader):
                attack_param = 1          
                x, y = x.to(device), y.to(device) 
                neurons = 0
                y_orig = torch.argmax(torch.from_numpy(base_model.predict(x.cpu().numpy())).to(device), dim = 1).squeeze()
                x_adv = attack.generate(x.cpu().numpy())
                model.zero_grad()
                y_adv = torch.argmax(torch.from_numpy(base_model.predict(x_adv)).to(device), dim = 1).squeeze()
                
                
                    
                cifar_adv.append(x_adv)
                labels.append(y_adv.cpu().numpy())    
                correct += (y == y_orig).sum().item()
                correct_adv += (y == y_adv).sum().item()
                model.zero_grad()
                run_total += x.size(0)
                if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)):
                    run_acc = correct/run_total
                    adv_run_acc = correct_adv/run_total
                    print(
                    f'Max_Evals: {max_evals[max_iter]}; ',
                    f'Train Acc: {round(run_acc, 2)}; ',
                    f'Adversarial Acc: {round(adv_run_acc, 2)}; ',
                    f'Time Elapsed: {time.time()-start}'
                    )
                break
            cifar_adv = np.reshape(np.asarray(cifar_adv),(-1,3,32,32))
            labels = np.asarray(labels).flatten()
            #confidence = np.reshape(np.asarray(confidence),(-1,10))
            np.save(load_path + 'HSJ_Norm_%d_Adversarial_Images_Eps_%d_Max_Evals_%d.npy'%(norm_str,int(test_eps[i]*1000),int(max_evals[max_iter]*1000)),
                cifar_adv)
            np.save(load_path + 'HSJ_Norm_%d_Labels_Eps_%d_Max_Evals_%d.npy'%(norm_str,int(test_eps[i]*1000),int(max_evals[max_iter]*1000)),
                labels)
            #np.save(load_path + 'HSJ_Confidence_Eps_%d_Attack_T_%d_Predict_T_%d.npy'%(int(max_evals[i]*1000),attack_timesteps,predict_timesteps),
            #    confidence)
            acc = correct/run_total
            acc_adv = correct_adv/run_total 
            print(
            f'Max Query: {max_evals[max_iter]}; ',
            f'Train Acc: {round(acc, 2)}; ',
            f'Adversarial Acc: {round(acc_adv, 2)}; ',
            f'Time Elapsed: {time.time()-start}'
            )
            accuracy_values[i] = test_eps[i],acc,acc_adv
            np.savetxt(load_path+'/HSJ_Norm_%d_Adversarial_Accuracy_Max_Evals_%d.txt'%(norm_str,int(max_evals[max_iter]*1000)),accuracy_values,fmt='%.6f')            
    
    return correct
def auto_attack(model, loader, T, criterion, device,load_path,arg_softmax, image_print, attack_step, predict_step,attack_norm):
    # Evaluate the model on a dataloader with T steps for the dynamics
    global attack_param
    model.eval()
    correct = 0
    start = time.time()
    phase = 'Attack Train' if loader.dataset.train else 'Attack Test'
    #eps = np.linspace(1,10,20)
    max_evals = [100,1000,10000]
    if(attack_norm==100):
        test_eps = np.logspace(np.log10(0.0001),np.log10(0.1),10)  
        #test_eps = np.insert(test_eps,0,1)
        test_eps = np.insert(test_eps,0,8/255)
        test_eps = np.sort(test_eps)
        #test_eps = np.array([2/255.0, 5/255.0, 8/255.0])
    else:

        test_eps = np.logspace(np.log10(0.05),np.log10(3),10) 
        test_eps = np.insert(test_eps,0,1)
        test_eps = np.insert(test_eps,0,0.005)
        test_eps = np.sort(test_eps)
        
    norm_str = attack_norm
    if(attack_norm==100):
        attack_norm=np.inf
    print(attack_norm)
    global predict_counter
    predict_counter = 0
    def predict_square(x):
        global predict_counter 
        predict_counter +=1
        x_c = torch.from_numpy(x).to(device)
        neurons = model.init_neurons(x_c.size(0), device)
        neurons = model(x_c,T=T)
        y_pred = F.softmax(model.synapses[-1](neurons[-1].view(x_c.size(0),-1)), dim = 1)
        return y_pred.detach().cpu().numpy()
    last_layer = 1
    accuracy_values = np.zeros((len(test_eps),3))
    mean = np.array([0.4914, 0.4822, 0.4465])[None,:,None,None]
    std=np.array([3*0.2023, 3*0.1994, 3*0.2010])[None,:,None,None]
    predict_timesteps = predict_step
    attack_timesteps = attack_step
    max_iter = 1
    try:
        accuracy_values = np.loadtxt(load_path+'/AA_Norm_%d_Square_Adversarial_Accuracy_Max_Evals_%d.txt'%(norm_str,int(max_evals[max_iter]*1000))) 
        idx = accuracy_values[:,0]==0
        
    except OSError:
        print('File Not Found')
    
    for i in range(2,len(test_eps)):	
        if(accuracy_values[i,0]!=0):
            continue
        attack_param = 1
        neurons = model.init_neurons(loader.batch_size, device)
        print(loader.batch_size)
        art_model = PyTorchClassifier(model, torch.nn.CrossEntropyLoss(), (3, 32, 32), 10, predict_timesteps, attack_timesteps, last_layer = last_layer, clip_values=(0,1),preprocessing=(mean,std))
        attack = AutoAttack(art_model, norm = attack_norm, eps = test_eps[i], batch_size=loader.batch_size)   
        correct = 0
        correct_adv = 0 
        if(image_print==0):
            mbs = loader.batch_size
            run_total = 0
            iter_per_epochs = math.ceil(len(loader.dataset)/mbs)
            cifar_adv, labels = [], []
            confidence = []
            batch_counter = 0
            for idx, (x, y) in enumerate(loader):
                attack_param = 1          
                x, y = x.to(device), y.to(device) 
                neurons = 0
                y_orig = torch.argmax(torch.from_numpy(art_model.predict(x.cpu().numpy())).to(device), dim = 1).squeeze()
                x_adv = attack.generate(x.cpu().numpy())
                model.zero_grad()
                y_adv = torch.argmax(torch.from_numpy(art_model.predict(x_adv)).to(device), dim = 1).squeeze()
                
                
                    
                cifar_adv.append(x_adv)
                labels.append(y_adv.cpu().numpy())    
                correct += (y == y_orig).sum().item()
                correct_adv += (y == y_adv).sum().item()
                model.zero_grad()
                run_total += x.size(0)
                if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)):
                    run_acc = correct/run_total
                    adv_run_acc = correct_adv/run_total
                    print(
                    f'Max_Evals: {max_evals[max_iter]}; ',
                    f'Train Acc: {round(run_acc, 2)}; ',
                    f'Adversarial Acc: {round(adv_run_acc, 2)}; ',
                    f'Time Elapsed: {time.time()-start}'
                    )
                batch_counter += 1
                if batch_counter == 1:
                    break
            cifar_adv = np.reshape(np.asarray(cifar_adv),(-1,3,32,32))
            labels = np.asarray(labels).flatten()
            #confidence = np.reshape(np.asarray(confidence),(-1,10))
            np.save(load_path + 'AA_Norm_%d_Square_Adversarial_Images_Eps_%d_Max_Evals_%d.npy'%(norm_str,int(test_eps[i]*1000),int(max_evals[max_iter]*1000)),
                cifar_adv)
            np.save(load_path + 'AA_Norm_%d_Square_Labels_Eps_%d_Max_Evals_%d.npy'%(norm_str,int(test_eps[i]*1000),int(max_evals[max_iter]*1000)),
                labels)
            np.save(load_path + 'AA_Norm_%d_Square_Confidence_Eps_%d_Attack_T_%d_Predict_T_%d.npy'%(norm_str,int(test_eps[i]*1000),attack_timesteps,predict_timesteps),
                confidence)
            acc = correct/run_total
            acc_adv = correct_adv/run_total 
            print(
            f'Max Query: {max_evals[max_iter]}; ',
            f'Train Acc: {round(acc, 2)}; ',
            f'Adversarial Acc: {round(acc_adv, 2)}; ',
            f'Time Elapsed: {time.time()-start}'
            )
            accuracy_values[i] = test_eps[i],acc,acc_adv
            np.savetxt(load_path+'/AA_Norm_%d_Square_Adversarial_Accuracy_Max_Evals_%d.txt'%(norm_str,int(max_evals[max_iter]*1000)),accuracy_values,fmt='%.6f')            
    
    return correct

def auto_attack_resume(model, loader, T, criterion, device,load_path,arg_softmax, image_print, attack_step, predict_step,attack_norm):
    # Evaluate the model on a dataloader with T steps for the dynamics
    global attack_param
    model.eval()
    correct = 0
    start = time.time()
    phase = 'Attack Train' if loader.dataset.train else 'Attack Test'
    #eps = np.linspace(1,10,20)
    max_evals = [100,1000,10000]
    if(attack_norm==100):
        test_eps = np.logspace(np.log10(0.0001),np.log10(0.1),10)  
        #test_eps = np.insert(test_eps,0,1)
        test_eps = np.insert(test_eps,0,8/255)
        test_eps = np.sort(test_eps)
        #test_eps = np.array([2/255.0, 5/255.0, 8/255.0])
    else:

        test_eps = np.logspace(np.log10(0.05),np.log10(3),10) 
        test_eps = np.insert(test_eps,0,1)
        test_eps = np.insert(test_eps,0,0.005)
        test_eps = np.sort(test_eps)
        
    norm_str = attack_norm
    if(attack_norm==100):
        attack_norm=np.inf
    print(attack_norm)
    global predict_counter
    predict_counter = 0
    def predict_square(x):
        global predict_counter 
        predict_counter +=1
        x_c = torch.from_numpy(x).to(device)
        neurons = model.init_neurons(x_c.size(0), device)
        neurons = model(x_c,T=T)
        y_pred = F.softmax(model.synapses[-1](neurons[-1].view(x_c.size(0),-1)), dim = 1)
        return y_pred.detach().cpu().numpy()
    last_layer = 1
    accuracy_values = np.zeros((len(test_eps),3))
    mean = np.array([0.4914, 0.4822, 0.4465])[None,:,None,None]
    std=np.array([3*0.2023, 3*0.1994, 3*0.2010])[None,:,None,None]
    predict_timesteps = predict_step
    attack_timesteps = attack_step
    max_iter = 1
    run_total = 200
    true_labels = np.load('/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/Natural_Images/Labels.npy')[:200]
    for i in range(2,len(test_eps)):	
            labels = np.load(load_path + 'AA_Norm_%d_Square_Labels_Eps_%d_Max_Evals_%d.npy'%(norm_str,int(test_eps[i]*1000),int(max_evals[max_iter]*1000)))
            print(labels.shape,true_labels.shape)
            correct = 200*0.885
            correct_adv = np.sum(labels==true_labels)

            acc = correct/run_total
            acc_adv = correct_adv/run_total 
            print(
            f'Max Query: {max_evals[max_iter]}; ',
            f'Train Acc: {round(acc, 2)}; ',
            f'Adversarial Acc: {round(acc_adv, 2)}; ',
            f'Time Elapsed: {time.time()-start}'
            )
            accuracy_values[i] = test_eps[i],acc,acc_adv
            np.savetxt(load_path+'/AA_Norm_%d_Square_Adversarial_Accuracy_Max_Evals_%d.txt'%(norm_str,int(max_evals[max_iter]*1000)),accuracy_values,fmt='%.6f')            
    
    return correct


def baseline_print(model, loader, T, criterion, device,load_path,arg_softmax):
    # Evaluate the model on a dataloader with T steps for the dynamics
    global attack_param
    model.eval()
    correct = 0
    start = time.time()
    
    
    
    mbs = loader.batch_size
    run_total = 0
    iter_per_epochs = math.ceil(len(loader.dataset)/mbs)
    cifar_adv, labels = [], []
    confidence = []
    for idx, (x, y) in enumerate(loader):    
        x, y = x.to(device), y.to(device) 
        neurons = model.init_neurons(loader.batch_size, device)
        neurons = model(x, y, neurons, T, beta=0, criterion=criterion)[-1]
        if not arg_softmax:
            pred = torch.argmax(neurons, dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
        else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
            pred = torch.argmax(F.softmax(model.synapses[-1](neurons.view(x.size(0),-1)), dim = 1), dim = 1).squeeze()

            confidence.append(model.synapses[-1](neurons.view(x.size(0),-1)).cpu().detach().numpy())
                
        cifar_adv.append(x.cpu().numpy())
        labels.append(y.cpu().numpy())    
        correct += (y == pred).sum().item()
        model.zero_grad()
        run_total += x.size(0)
        if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)):
            run_acc = correct/run_total
            adv_run_acc = correct/run_total
            print(
            f'Idx: {idx}; ',
            f'Train Acc: {round(run_acc, 2)}; ',
            f'Adversarial Acc: {round(adv_run_acc, 2)}; ',
            f'Time Elapsed: {time.time()-start}'
            )
    cifar_adv = np.reshape(np.asarray(cifar_adv),(-1,3,32,32))
    labels = np.asarray(labels).flatten()
    confidence = np.reshape(np.asarray(confidence),(-1,10))
    np.save(load_path + 'Original_Images.npy',
        cifar_adv)
    np.save(load_path + 'Original_Labels.npy',
        labels)
    np.save(load_path + 'Original_Confidence_Predict_T_%d.npy'%(T),
        confidence)
        
    return

def corruptions(model, loader, T, device, load_path, arg_softmax):
    model.eval()
    correct = 0
    start = time.time()
    corruption_accuracy = []
    norm_layer = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (3*0.2023, 3*0.1994, 3*0.2010))
    for p in ['gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur',
                'spatter', 'brightness' ,'speckle_noise', 'gaussian_blur', 'snow',
                'contrast', 'defocus_blur','elastic_transform','fog','glass_blur','impulse_noise','jpeg_compression',
                'pixelate','saturate','frost']:
        c_p_dir = '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/vision-greg3/CIFAR-10-C'
        label_dir = '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/vision-greg3/CIFAR-10-C'
        labels = torch.from_numpy(np.load(os.path.join(label_dir, 'labels.npy')))
        dataset = np.load(os.path.join(c_p_dir, p + '.npy'))
        dataset = torch.from_numpy(np.transpose(dataset,(0,3,1,2)))/255.
        ood_data = torch.utils.data.TensorDataset(dataset, labels)
        mbs = 100
        loader = torch.utils.data.DataLoader(
            ood_data, batch_size=mbs, shuffle=False, num_workers=2, pin_memory=True)
        
        
        avg_acc = []
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            neurons = model(norm_layer(data))[-1]
            if not arg_softmax:
                pred = torch.argmax(neurons, dim=1).squeeze()  # in this cas prediction is done directly on the last (output) layer of neurons
            else: # prediction is done as a readout of the penultimate layer (output is not part of the system)
                pred = torch.argmax(F.softmax(model.synapses[-1](neurons.view(data.size(0),-1)), dim = 1), dim = 1).squeeze()

            correct = (target == pred).cpu().numpy()
            avg_acc.append(correct)
        
        avg_acc = np.asarray(avg_acc)
        avg_acc = np.reshape(avg_acc,(5,(avg_acc.shape[1]*avg_acc.shape[0])//5))    
        avg_acc = avg_acc.mean(axis=1)
        corruption_accuracy.append(avg_acc)
    corruption_accuracy = np.asarray(corruption_accuracy)
    np.savetxt(load_path+'/Corruption_Accuracy.txt', corruption_accuracy,fmt='%.6f')