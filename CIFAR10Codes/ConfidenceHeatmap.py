import numpy as np
import torchvision
import torch.nn.functional as F
from scipy.special import softmax
import os
from datetime import datetime
import time
import math
from data_utils import *
import glob
from itertools import repeat
from torch.nn.parameter import Parameter
import collections
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import re
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch
from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.estimators.classification import PyTorchClassifier
from art.estimators.classification import BlackBoxClassifierNeuralNetwork
from art.attacks.evasion.square_attack import SquareAttack
matplotlib.use('Agg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.weight": "bold"
})
results_path = '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/'
def find_params():
    param_list = []
    dname = 'EP/cel/2023-06-14/17-04-29_gpu0/'
    for filename in glob.glob(results_path+dname+'White_Last_Layer_Confidence_*Attack_T_70_*.npy'):
        fname = filename.replace(results_path+dname,'')
        param_list.append(re.findall(r'\d+',fname))
    return np.asarray(param_list,dtype=int)
params = find_params()
eps = params[:,0]
Attack_T = params[0,1]
Predict_T = params[0,2]
eps = np.sort(eps)/1000.0
def get_base_index(fname):
    confidences = np.load(fname)
    confidences = softmax(confidences,axis=1)
    predictions = np.argmax(confidences,axis=1)
    correct_idx = labels==predictions
    return correct_idx
def get_confidence_base_comparison(fname,correct_idx):
    confidences = np.load(fname)
    confidences = softmax(confidences,axis=1)
    predictions = np.argmax(confidences,axis=1)
    incorrect_idx = correct_idx
    label_subset = labels[incorrect_idx]
    confidence_subset = confidences[incorrect_idx]
    max_confidence = np.max(confidence_subset,axis=1)
    mean = np.mean(max_confidence)#1-sum(incorrect_idx)/len(incorrect_idx)
    std = np.std(max_confidence)
    confidence_matrix = np.zeros((10,10))
    for j in range(10):
        label_idx = label_subset==j
        confidence_subsubset = confidence_subset[label_idx]
        confidence_matrix[j] = np.mean(confidence_subsubset,axis=0)
    return confidence_matrix, mean, std
def get_confidence_matrix(fname):
    confidences = np.load(fname)
    confidences = softmax(confidences,axis=1)
    predictions = np.argmax(confidences,axis=1)
    incorrect_idx = labels!=predictions
    label_subset = labels[incorrect_idx]
    confidence_subset = confidences[incorrect_idx]
    max_confidence = np.max(confidence_subset,axis=1)
    mean = np.mean(max_confidence)#1-sum(incorrect_idx)/len(incorrect_idx)
    std = np.std(max_confidence)
    confidence_matrix = np.zeros((10,10))
    for j in range(10):
        label_idx = label_subset==j
        confidence_subsubset = confidence_subset[label_idx]
        confidence_matrix[j] = np.mean(confidence_subsubset,axis=0)
    return confidence_matrix, mean, std
def get_fname(f_iter,eps):
    if f_iter == 0:
        fname = results_path+f'EP/cel/2023-06-14/17-04-29_gpu0/White_Last_Layer_Confidence_Eps_{eps}_Attack_T_70_Predict_T_250.npy'
    if f_iter == 1:
        fname = results_path+f'BaselineCNN/cel/Jupyter/MTModel/Confidence_Norm_2_Train_Eps_0_Test_Eps_{eps}_Seed_11.npy'
    if f_iter == 2:
        fname = results_path+f'BaselineCNN/cel/Jupyter/MTModel/Confidence_Norm_2_Train_Eps_1000_Test_Eps_{eps}_Seed_11.npy'
    if f_iter == 3:
        fname = f'/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/ViT-CIFAR/Norm_2_Confidence_Eps_{eps}_Patch_4_Augment_1.npy'
    return fname
labels = np.load('/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/Natural_Images/Labels.npy')
labeldict = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
ticks = np.arange(10)
a=2
titledict = ['EP CNN','CNN',r'CNN $(\epsilon=1.0)$','ViT' ]
base_indices = []
for i in range(4):
    fname = get_fname(i,int(eps[0]*1000))
    base_indices.append(get_base_index(fname))
base_indices = np.asarray(base_indices)
confidence_max = np.zeros((4,len(eps),2))
for i in range(len(eps)):
    fig,ax = plt.subplots(1,4,figsize=(24,8))
    for file_iter in range(4):
        fname = get_fname(file_iter,int(eps[i]*1000))
        confidence_matrix, mean, std = get_confidence_base_comparison(fname,base_indices[file_iter])
        #confidence_max[file_iter][i] = mean, std 
        confidence_max[file_iter][i] = np.mean(np.max(confidence_matrix,axis=1)), np.std(np.max(confidence_matrix,axis=1))
        pos = ax[file_iter].imshow(confidence_matrix, cmap='hot',vmax = 0.8,vmin=0)
        ax[file_iter].set_xticks(ticks)
        ax[file_iter].set_xticklabels(labeldict,rotation = 45,fontsize=15)
        ax[file_iter].set_yticks(ticks)
        ax[file_iter].set_yticklabels(labeldict,rotation = 45,fontsize=15)
        ax[file_iter].set_title(titledict[file_iter],fontsize=25)
        """ ax[file_iter].set_xlabel('Misclassified Label',fontsize=25)
        ax[file_iter].set_ylabel('Correct Label',fontsize=25)"""
        if(file_iter==3):
            divider = make_axes_locatable(ax[file_iter])
            cax = divider.append_axes("right", size="5%", pad=0.08)
            cb = plt.colorbar( pos, ax=ax[file_iter], cax=cax )
    fig.suptitle(r'$l_2$ Perturbation $\epsilon=%.2f$'%eps[i],fontsize=30)
    plt.tight_layout()
    plt.savefig('Confidence_Matrices/Confidence_Matrix_Eps_%d.png'%int(eps[i]*1000))
fig,ax = plt.subplots(1,1,figsize=(10,8))
for file_iter in range(4):
    ax.plot(eps,confidence_max[file_iter][:,0],lw=4,label=titledict[file_iter])
ax.set_xlabel(r'$l_2$ Perturbation $\epsilon$',fontsize=30)
ax.set_ylabel('Confidence',fontsize=30)
ax.set_xscale('log')
ax.xaxis.set_tick_params(labelsize=25)
ax.yaxis.set_tick_params(labelsize=25)
ax.legend(loc=1,fontsize=30)
ax.grid()
[x.set_linewidth(1.5) for x in ax.spines.values()]
plt.tight_layout()
plt.savefig('Confidence_Matrices/Misclassification_Confidence.png')

