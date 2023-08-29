import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cmx
import matplotlib.colors as mcol
import glob
import torch.nn.functional as F
import re
import torch
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.weight": "bold"
})

dname = os.getcwd()
def find_files(file_loc_list):
    param_list = []
    for i in range(len(file_loc_list)):
        for filename in glob.glob(dname+file_loc_list[i]+'Adversarial_Accuracy_A*.txt'):
            fname = filename.replace(dname+file_loc_list[i],'')
            param_list.append(re.findall(r'\d+',fname))
    param_list = np.asarray(param_list,dtype=int)
    param_list = np.unique(param_list,axis=0)
    return param_list
def find_files(file_loc_list):
    param_list = []
    for i in range(len(file_loc_list)):
        for filename in glob.glob(dname+file_loc_list[i]+'Adversarial_Accuracy_A*.txt'):
            fname = filename.replace(dname+file_loc_list[i],'')
            param_list.append(re.findall(r'\d+',fname))
    param_list = np.asarray(param_list,dtype=int)
    param_list = np.unique(param_list,axis=0)
    return param_list
def find_filesll(file_loc_list):
    param_list = []
    for i in range(len(file_loc_list)):
        for filename in glob.glob(dname+file_loc_list[i]+'White_Last_Layer_Adversarial_Accuracy_A*.txt'):
            fname = filename.replace(dname+file_loc_list[i],'')
            param_list.append(re.findall(r'\d+',fname))
    param_list = np.asarray(param_list,dtype=int)
    param_list = np.unique(param_list,axis=0)
    return param_list
def find_filesll_advtrain(file_loc_list):
    param_list = []
    for i in range(len(file_loc_list)):
        for filename in glob.glob(dname+file_loc_list[i]+'White_Last_Layer_Norm_2_Adversarial_Accuracy_A*.txt'):
            fname = filename.replace(dname+file_loc_list[i],'')
            param_list.append(re.findall(r'\d+',fname))
    param_list = np.asarray(param_list,dtype=int)
    param_list = np.unique(param_list,axis=0)
    return param_list[:,[1,2]]

def plot_characteristic_time(ax):
    
    file_loc = ['/results/EP/mse/2023-05-22/12-31-48_gpu0/',
                '/results/EP/mse/2023-05-22/12-28-59_gpu0/',
                '/results/EP/cel/2023-05-23/16-52-21_gpu0/',
                '/results/EP/cel/2023-05-23/15-40-03_gpu0/',
                '/results/BPTT/mse/2023-05-22/13-52-18_gpu0/',
                '/results/BPTT/cel/2023-05-30/15-31-49_gpu0/',
                '/results/BaselineCNN/cel/Jupyter/','/results/BaselineCNN/cel/Jupyter/']
    labels = ['Alg: EP With Symmetric Gradient, Loss: MSE',
              'Alg: EP With Random Sign, Loss: MSE',
              'Alg: EP With Symmetric Gradient (Softmax), Loss: CE',
              'Alg: EP With Symmetric Gradient, Loss: CE',
              'Alg: BPTT, Loss: MSE',
              'Alg: BPTT (Softmax), Loss: CE',
              'Vanilla CNN (HardSigmoid Activations), Loss: CE','Vanilla CNN (ReLU Activations), Loss: CE']
    for file_iter in range(2,len(file_loc)):
        if(file_iter==3):
            continue
        filename = file_loc[file_iter]
        if(file_iter<6):

            
            characteristic_time = np.loadtxt(dname+filename+'Characteristic_Time.txt')
            ax.plot(characteristic_time[:,0],characteristic_time[:,1],label=labels[file_iter])
        elif(file_iter==6):
            characteristic_time = np.loadtxt(dname+filename+'Adversarial_Accuracy_UnNormalized_HardSigmoid.txt')
            print(characteristic_time[0][1])
            ax.plot(np.arange(260),np.ones(260)*characteristic_time[0][1],label=labels[file_iter])
        elif(file_iter==7):
            characteristic_time = np.loadtxt(dname+filename+'Adversarial_Accuracy_UnNormalized_Relu.txt')
            ax.plot(np.arange(260),np.ones(260)*characteristic_time[0][1],label=labels[file_iter])
            print(characteristic_time[0][1])

    ax.set_xlabel('Free Phase Iterations',fontsize=25)
    ax.set_ylabel('Test Accuracy',fontsize=25)
    ax.set_xlim([1,30])
    ax.set_ylim([0.05,1])
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.legend(loc=4,fontsize=15)
    ax.grid()
    [x.set_linewidth(1.5) for x in ax.spines.values()]
def plot_adversarial_accuracy(ax):
    
    file_loc = [['/results/EP/cel/2023-06-14/17-04-29_gpu0/Adversarial_Accuracy_UnNormalized.txt',
                '/results/EP/cel/2023-06-14/17-06-02_gpu0/Adversarial_Accuracy_UnNormalized.txt',
                '/results/EP/cel/2023-06-14/17-06-08_gpu0/Adversarial_Accuracy_UnNormalized.txt',
                '/results/EP/cel/2023-06-14/17-06-24_gpu0/Adversarial_Accuracy_UnNormalized.txt',
                '/results/EP/cel/2023-06-15/00-48-27_gpu0/Adversarial_Accuracy_UnNormalized.txt'],
                ['/results/BPTT/cel/2023-05-30/15-31-49_gpu0/Adversarial_Accuracy_UnNormalized.txt'],
                ['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_hardsigmoid.txt'],['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_relu.txt']]
    labels = ['Alg: EP With Symmetric Gradient, Loss: CE',
              'Alg: BPTT, Loss: CE',
              'Vanilla CNN (HardSigmoid), Loss: CE','Vanilla CNN (ReLU), Loss: CE']
    colors = ['k','m','g','c']
    for label_iter in range(len(labels)):
        if(label_iter==1):
            continue
        adversarial_accuracies = []
        for filename in file_loc[label_iter]:
            try:
                adversarial_accuracy = np.loadtxt(dname+filename)
                adversarial_accuracies.append(adversarial_accuracy)
            except OSError:
                continue 
        if(len(adversarial_accuracies)!=0):
            adversarial_accuracy_mean = np.mean(np.asarray(adversarial_accuracies),axis=0)
            adversarial_accuracy_std = np.std(np.asarray(adversarial_accuracies),axis=0)
            if(label_iter==0):
                ax.errorbar(adversarial_accuracy_mean[:,0],adversarial_accuracy_mean[:,2], yerr = adversarial_accuracy_std[:,2],label=labels[label_iter],lw=2,c=colors[label_iter])
            else:
                ax.plot(adversarial_accuracy_mean[:,0],adversarial_accuracy_mean[:,2],'.-',label=labels[label_iter],lw=2,c=colors[label_iter])
    if(len(adversarial_accuracies)!=0):
        ax.set_xlabel(r'Perturbation $\epsilon$',fontsize=25)
        ax.yaxis.tick_right()
        ax.set_ylabel('Test Accuracy',fontsize=25)
        #ax.set_xlim([1,100])
        ax.set_xscale('log')
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax.set_ylim([0.05,0.95])
        ax.set_xlim([0.05,3])
        ax.legend(loc=3,fontsize=15)
        ax.grid()
        [x.set_linewidth(1.5) for x in ax.spines.values()]
def plot_adversarial_accuracy2(ax):
    
    file_loc = [['/results/EP/cel/2023-06-14/17-04-29_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-02_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-08_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-24_gpu0/',
                '/results/EP/cel/2023-06-15/00-48-27_gpu0/'],
                ['/results/BPTT/cel/2023-06-19/17-19-25_gpu0/',
                 '/results/BPTT/cel/2023-06-19/18-28-16_gpu0/',
                 '/results/BPTT/cel/2023-06-19/19-10-16_gpu0/',
                 '/results/BPTT/cel/2023-06-19/19-16-24_gpu0/',
                 '/results/BPTT/cel/2023-06-19/17-17-24_gpu0/'],
                ['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_hardsigmoid.txt'],['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_relu.txt']]
    labels = ['EP',
              'BPTT',
              'Vanilla CNN (HardSigmoid), Loss: CE','Vanilla CNN (ReLU), Loss: CE']
    colors = ['y','m','g','c']
    fname_template = 'Adversarial_Accuracy_Attack_T_%d_Predict_T_%d.txt'
    for label_iter in range(len(labels)):
        if(label_iter>-1):
            continue
        param_list = find_files(file_loc[label_iter])
        param_list = param_list[param_list[:,0]==29]

        param_list = param_list[param_list[:,1]==250]
        #param_list = param_list[::2,]
        cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["k","c"])
        cNorm = mcol.Normalize(vmin=min(param_list[:,0]), vmax=max(param_list[:,0]))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        for params in param_list:
            
            print(labels[label_iter],params)
            adversarial_accuracies = []
            for file in file_loc[label_iter]:
                try:
                    adversarial_accuracy = np.loadtxt(dname+file+fname_template%(params[0],params[1]))
                    adversarial_accuracies.append(adversarial_accuracy)
                except OSError:
                    continue 
            if(len(adversarial_accuracies)!=0):
                adversarial_accuracies = np.asarray(adversarial_accuracies)
                adversarial_accuracies = adversarial_accuracies.reshape((adversarial_accuracies.shape[0]*adversarial_accuracies.shape[1],adversarial_accuracies.shape[-1]))
                idx = np.argwhere(adversarial_accuracies[:,0]!=0).flatten()
                adversarial_accuracies = adversarial_accuracies[idx]
                epsilons = np.unique(adversarial_accuracies[:,0])
                adversarial_accuracy_stats = np.zeros((len(epsilons),3))
                
                for eps_iter in range(len(epsilons)):
                    subset = adversarial_accuracies[adversarial_accuracies[:,0]==epsilons[eps_iter]]
                    adversarial_accuracy_stats[eps_iter] = epsilons[eps_iter], np.mean(subset[:,2]), np.std(subset[:,2])
                if(label_iter<2):
                    if(label_iter==0):
                        #ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1], yerr = adversarial_accuracy_stats[:,2],
                        #            label='%s $T_{attack}=%d$'%(labels[label_iter],params[0]), lw=2,c=scalarMap.to_rgba(params[0]))
                         ax.plot(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],'.-',
                                    label='%s $T_{attack}=%d$'%(labels[label_iter],params[0]), lw=2,c=scalarMap.to_rgba(params[0]))
                    else:
                        ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1], yerr = adversarial_accuracy_stats[:,2],label='%s $T_{attack}=%d$'%(labels[label_iter],params[0]),lw=2,c=colors[label_iter])
    #if(len(adversarial_accuracies)!=0):
    ax.set_xlabel(r'$l_2$ Perturbation $\epsilon$',fontsize=25)
    ax.set_ylabel('Test Accuracy',fontsize=25)
    #ax.set_xlim([1,100])
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_ylim([0.05,0.95])
    ax.set_xlim([0.05,3.1])
    ax.legend(loc=3,fontsize=15)
    ax.grid()
    [x.set_linewidth(1.5) for x in ax.spines.values()]
    return 0#np.unique(param_list[:,0])
def plot_adversarial_accuracyVanilla(ax):
    
    file_loc = [['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_hardsigmoid.txt'],['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_relu.txt']]
    labels = ['CNN','CNN']
    colors = ['g','c']
    for label_iter in range(1,2):
        adversarial_accuracies = []
        for file in file_loc[label_iter]:
            try:
                adversarial_accuracy = np.loadtxt(dname+file)
                adversarial_accuracies.append(adversarial_accuracy)
            except OSError:
                continue 
        if(len(adversarial_accuracies)!=0):
            adversarial_accuracies = np.asarray(adversarial_accuracies)
            adversarial_accuracies = adversarial_accuracies.reshape((adversarial_accuracies.shape[0]*adversarial_accuracies.shape[1],adversarial_accuracies.shape[-1]))
            idx = np.argwhere(adversarial_accuracies[:,0]!=0).flatten()
            adversarial_accuracies = adversarial_accuracies[idx]
            epsilons = np.unique(adversarial_accuracies[:,0])
            adversarial_accuracy_stats = np.zeros((len(epsilons),3))
            
            for eps_iter in range(len(epsilons)):
                subset = adversarial_accuracies[adversarial_accuracies[:,0]==epsilons[eps_iter]]
                adversarial_accuracy_stats[eps_iter] = epsilons[eps_iter], np.mean(subset[:,2]), np.std(subset[:,2])
            ax.plot(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],'.-',
                                label=labels[label_iter], lw=2)
def plot_adversarial_accuracyVanillaInf(ax):
    
    file_loc = [['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Norm_%d_Accuracy_Model_hardsigmoid_Seed_11.txt'],['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Norm_%d_Accuracy_Model_relu_Seed_11.txt']]
    labels = ['VCNN (HardSigmoid)','VCNN (ReLU)']
    colors = ['g','c']
    for label_iter in range(len(labels)):
        adversarial_accuracies = []
        for file in file_loc[label_iter]:
            try:
                adversarial_accuracy = np.loadtxt(dname+file%100)
                adversarial_accuracies.append(adversarial_accuracy)
            except OSError:
                continue 
        if(len(adversarial_accuracies)!=0):
            adversarial_accuracies = np.asarray(adversarial_accuracies)
            adversarial_accuracies = adversarial_accuracies.reshape((adversarial_accuracies.shape[0]*adversarial_accuracies.shape[1],adversarial_accuracies.shape[-1]))
            idx = np.argwhere(adversarial_accuracies[:,0]!=0).flatten()
            adversarial_accuracies = adversarial_accuracies[idx]
            epsilons = np.unique(adversarial_accuracies[:,0])
            adversarial_accuracy_stats = np.zeros((len(epsilons),3))
            
            for eps_iter in range(len(epsilons)):
                subset = adversarial_accuracies[adversarial_accuracies[:,0]==epsilons[eps_iter]]
                adversarial_accuracy_stats[eps_iter] = epsilons[eps_iter], np.mean(subset[:,2]), np.std(subset[:,2])
            ax.plot(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],'.-',
                                label=labels[label_iter], lw=2)
def plot_adversarial_trained_accuracy2ll(ax):
    
    file_loc = [['/results/EP_adv/cel/2023-07-01/19-47-05_gpu0/']]
    labels = ['Adv EP']
    colors = ['y','m','g','c']
    fname_template = 'White_Last_Layer_Norm_2_Adversarial_Accuracy_Attack_T_%d_Predict_T_%d.txt'
    for label_iter in range(len(labels)):
        if(label_iter>0):
            continue
        param_list = find_filesll_advtrain(file_loc[label_iter])
        print(file_loc[label_iter],param_list)
        #param_list = param_list[param_list[:,0]==10]

        param_list = param_list[param_list[:,1]==250]
        #param_list = param_list[::2,]
        cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["k","c"])
        cNorm = mcol.LogNorm(vmin=min(param_list[:,0]), vmax=max(param_list[:,0]))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        for params in param_list:
            
            print(labels[label_iter],params)
            adversarial_accuracies = []
            for file in file_loc[label_iter]:
                try:
                    ##print(dname+file+fname_template%(params[0],params[1]))
                    adversarial_accuracy = np.loadtxt(dname+file+fname_template%(params[0],params[1]))
                    adversarial_accuracies.append(adversarial_accuracy)
                except OSError:
                    continue 
            
            if(len(adversarial_accuracies)!=0):
                adversarial_accuracies = np.asarray(adversarial_accuracies)
                adversarial_accuracies = adversarial_accuracies.reshape((adversarial_accuracies.shape[0]*adversarial_accuracies.shape[1],adversarial_accuracies.shape[-1]))
                idx = np.argwhere(adversarial_accuracies[:,0]!=0).flatten()
                adversarial_accuracies = adversarial_accuracies[idx]
                epsilons = np.unique(adversarial_accuracies[:,0])
                adversarial_accuracy_stats = np.zeros((len(epsilons),3))
                
                for eps_iter in range(len(epsilons)):
                    subset = adversarial_accuracies[adversarial_accuracies[:,0]==epsilons[eps_iter]]
                    adversarial_accuracy_stats[eps_iter] = epsilons[eps_iter], np.mean(subset[:,2]), np.std(subset[:,2])
                if(label_iter<2):
                    if(label_iter==0):
                        #ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1], yerr = adversarial_accuracy_stats[:,2],
                        #            label='%s $T_{attack}=%d$'%(labels[label_iter],params[0]), lw=2,c=scalarMap.to_rgba(params[0]))
                        ax.plot(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],'.-',
                                    label='%s'%(labels[label_iter]), lw=2,c=scalarMap.to_rgba(params[0]))
                    else:
                        ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1], yerr = adversarial_accuracy_stats[:,2],label='%s $T_{attack}=%d$'%(labels[label_iter],params[0]),lw=2,c=colors[label_iter])
    """ if(len(adversarial_accuracies)!=0):
        ax.set_xlabel(r'Perturbation $\epsilon$',fontsize=25)
        ax.set_ylabel('Test Accuracy',fontsize=25)
        #ax.set_xlim([1,100])
        ax.set_xscale('log')
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax.set_ylim([0.05,0.95])
        ax.set_xlim([0.05,3.1])
        ax.legend(loc=3,fontsize=15)
        ax.grid()
        [x.set_linewidth(1.5) for x in ax.spines.values()] """
    return np.unique(param_list[:,0])               
def plot_adversarial_accuracy2ll(ax):
    
    file_loc = [['/results/EP/cel/2023-06-14/17-04-29_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-02_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-08_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-24_gpu0/',
                '/results/EP/cel/2023-06-15/00-48-27_gpu0/'],
                ['/results/BPTT/cel/2023-06-19/17-19-25_gpu0/',
                 '/results/BPTT/cel/2023-06-19/18-28-16_gpu0/',
                 '/results/BPTT/cel/2023-06-19/19-10-16_gpu0/',
                 '/results/BPTT/cel/2023-06-19/19-16-24_gpu0/',
                 '/results/BPTT/cel/2023-06-19/17-17-24_gpu0/'],
                ['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_hardsigmoid.txt'],['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_relu.txt']]
    labels = ['EP',
              'BPTT',
              'VCNN (HardSigmoid)','VCNN (ReLU)']
    colors = ['y','m','g','c']
    fname_template = 'White_Last_Layer_Adversarial_Accuracy_Attack_T_%d_Predict_T_%d.txt'
    for label_iter in range(len(labels)):
        if(label_iter>0):
            continue
        param_list = find_filesll(file_loc[label_iter])
        
        param_list = param_list[param_list[:,0]==10]

        param_list = param_list[param_list[:,1]==250]
        #param_list = param_list[::2,]
        cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["k","c"])
        cNorm = mcol.LogNorm(vmin=min(param_list[:,0]), vmax=max(param_list[:,0]))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        for params in param_list:
            
            print(labels[label_iter],params)
            adversarial_accuracies = []
            for file in file_loc[label_iter]:
                try:
                    ##print(dname+file+fname_template%(params[0],params[1]))
                    adversarial_accuracy = np.loadtxt(dname+file+fname_template%(params[0],params[1]))
                    adversarial_accuracies.append(adversarial_accuracy)
                except OSError:
                    continue 
            
            if(len(adversarial_accuracies)!=0):
                adversarial_accuracies = np.asarray(adversarial_accuracies)
                adversarial_accuracies = adversarial_accuracies.reshape((adversarial_accuracies.shape[0]*adversarial_accuracies.shape[1],adversarial_accuracies.shape[-1]))
                idx = np.argwhere(adversarial_accuracies[:,0]!=0).flatten()
                adversarial_accuracies = adversarial_accuracies[idx]
                epsilons = np.unique(adversarial_accuracies[:,0])
                adversarial_accuracy_stats = np.zeros((len(epsilons),3))
                
                for eps_iter in range(len(epsilons)):
                    subset = adversarial_accuracies[adversarial_accuracies[:,0]==epsilons[eps_iter]]
                    adversarial_accuracy_stats[eps_iter] = epsilons[eps_iter], np.mean(subset[:,2]), np.std(subset[:,2])
                if(label_iter<2):
                    if(label_iter==0):
                        #ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1], yerr = adversarial_accuracy_stats[:,2],
                        #            label='%s $T_{attack}=%d$'%(labels[label_iter],params[0]), lw=2,c=scalarMap.to_rgba(params[0]))
                        ax.plot(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],'--',
                                    label='%s'%(labels[label_iter]), lw=2,c=scalarMap.to_rgba(params[0]))
                    else:
                        ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1], yerr = adversarial_accuracy_stats[:,2],label='%s $T_{attack}=%d$'%(labels[label_iter],params[0]),lw=2,c=colors[label_iter])
    """ if(len(adversarial_accuracies)!=0):
        ax.set_xlabel(r'Perturbation $\epsilon$',fontsize=25)
        ax.set_ylabel('Test Accuracy',fontsize=25)
        #ax.set_xlim([1,100])
        ax.set_xscale('log')
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax.set_ylim([0.05,0.95])
        ax.set_xlim([0.05,3.1])
        ax.legend(loc=3,fontsize=15)
        ax.grid()
        [x.set_linewidth(1.5) for x in ax.spines.values()] """
    return np.unique(param_list[:,0])
def plot_adversarial_accuracyinfll(ax):
    
    file_loc = [['/results/EP/cel/2023-06-14/17-04-29_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-02_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-08_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-24_gpu0/',
                '/results/EP/cel/2023-06-15/00-48-27_gpu0/'],
                ['/results/BPTT/cel/2023-06-19/17-19-25_gpu0/',
                 '/results/BPTT/cel/2023-06-19/18-28-16_gpu0/',
                 '/results/BPTT/cel/2023-06-19/19-10-16_gpu0/',
                 '/results/BPTT/cel/2023-06-19/19-16-24_gpu0/',
                 '/results/BPTT/cel/2023-06-19/17-17-24_gpu0/'],
                ['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_hardsigmoid.txt'],['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_relu.txt']]
    labels = ['EP',
              'BPTT',
              'VCNN (HardSigmoid)','VCNN (ReLU)']
    colors = ['y','m','g','c']
    fname_template = 'White_Last_Layer_Norm_%d_Adversarial_Accuracy_Attack_T_%d_Predict_T_%d.txt'
    for label_iter in range(len(labels)):
        if(label_iter>0):
            continue
        param_list = find_filesll(file_loc[label_iter])
        
        #param_list = param_list[param_list[:,0]]

        param_list = param_list[param_list[:,1]==250]
        #param_list = param_list[::2,]
        cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["k","c"])
        cNorm = mcol.LogNorm(vmin=min(param_list[:,0]), vmax=max(param_list[:,0]))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        for params in param_list:
            
            print(labels[label_iter],params)
            adversarial_accuracies = []
            for file in file_loc[label_iter]:
                try:
                    ##print(dname+file+fname_template%(params[0],params[1]))
                    adversarial_accuracy = np.loadtxt(dname+file+fname_template%(100,params[0],params[1]))
                    adversarial_accuracies.append(adversarial_accuracy)
                except OSError:
                    continue 
            
            if(len(adversarial_accuracies)!=0):
                adversarial_accuracies = np.asarray(adversarial_accuracies)
                adversarial_accuracies = adversarial_accuracies.reshape((adversarial_accuracies.shape[0]*adversarial_accuracies.shape[1],adversarial_accuracies.shape[-1]))
                idx = np.argwhere(adversarial_accuracies[:,0]!=0).flatten()
                adversarial_accuracies = adversarial_accuracies[idx]
                epsilons = np.unique(adversarial_accuracies[:,0])
                adversarial_accuracy_stats = np.zeros((len(epsilons),3))
                
                for eps_iter in range(len(epsilons)):
                    subset = adversarial_accuracies[adversarial_accuracies[:,0]==epsilons[eps_iter]]
                    adversarial_accuracy_stats[eps_iter] = epsilons[eps_iter], np.mean(subset[:,2]), np.std(subset[:,2])
                if(label_iter<2):
                    if(label_iter==0):
                        #ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1], yerr = adversarial_accuracy_stats[:,2],
                        #            label='%s $T_{attack}=%d$'%(labels[label_iter],params[0]), lw=2,c=scalarMap.to_rgba(params[0]))
                        ax.plot(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],'--',
                                    label='%s $T_{attack}=%d$'%(labels[label_iter],params[0]), lw=2,c=scalarMap.to_rgba(params[0]))
                    else:
                        ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1], yerr = adversarial_accuracy_stats[:,2],label='%s $T_{attack}=%d$'%(labels[label_iter],params[0]),lw=2,c=colors[label_iter])
    
    ax.set_xlabel(r'$l_\infty$ Perturbation $\epsilon$',fontsize=25)
    ax.set_ylabel('Test Accuracy',fontsize=25)
    #ax.set_xlim([1,100])
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_ylim([0.05,0.95])
    ax.set_xlim([0.0001,0.11])
    ax.legend(loc=3,fontsize=15)
    ax.grid()
    [x.set_linewidth(1.5) for x in ax.spines.values()]
    return np.unique(param_list[:,0])
def plot_confidence(ax):
    eps = np.logspace(np.log10(0.05),np.log10(3),10)
    eps = np.insert(eps,0,1)
    eps = np.sort(eps)
    file_loc = [['/results/EP/cel/2023-06-14/17-04-29_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-02_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-08_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-24_gpu0/',
                '/results/EP/cel/2023-06-15/00-48-27_gpu0/'],
                ['/results/BPTT/cel/2023-06-19/17-19-25_gpu0/',
                 '/results/BPTT/cel/2023-06-19/18-28-16_gpu0/',
                 '/results/BPTT/cel/2023-06-19/19-10-16_gpu0/',
                 '/results/BPTT/cel/2023-06-19/19-16-24_gpu0/',
                 '/results/BPTT/cel/2023-06-19/17-17-24_gpu0/'],
                ['/results/BaselineCNN/cel/Jupyter/MTModel/']]
    labels = ['EP',
              'BPTT',
              'Vanilla CNN']
    colors = ['y','m','g','c']
    train_eps = [0.05,0.5,0.75,1]
    cm = plt.get_cmap('jet')
    cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    cNorm = mcol.Normalize(vmin=min(train_eps), vmax=max(train_eps))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    for label_iter in range(len(labels)):
        if(label_iter==1):
            continue        
        x,y,y_errorbar = [], [], []
        confidence_eps = np.zeros((len(eps),10,3))
        if(label_iter==0):
            for eps_iter in range(len(eps)):  
                for file in file_loc[label_iter]:
                    try:
                        confidence_values = np.load(dname+file+'Confidence_Eps_%d_Attack_T_10_Predict_T_250.npy'%(int(eps[eps_iter]*1000)))
                        original_labels = np.load(dname+file+'Original_Labels.npy')
                    except OSError:
                        continue 
                    pred = torch.argmax(F.softmax(torch.from_numpy(confidence_values),dim=1),dim=1).squeeze()
                    pred = pred.numpy()
                    idx = pred==original_labels
                    confidence_values = confidence_values[idx]
                    confidence = torch.max(F.softmax(torch.from_numpy(confidence_values),dim=1),dim=1)
                    confidence_argmax = confidence.indices.squeeze().numpy()
                    confidence = confidence.values.squeeze().numpy()
                    for class_type in range(10):
                        idx = confidence_argmax==class_type
                        subset = confidence[idx]
                    
                        confidence_eps[eps_iter][class_type][0] = np.mean(subset)
                        confidence_eps[eps_iter][class_type][1] = np.std(subset)
                        confidence_eps[eps_iter][class_type][2] = sum(idx)
            
                y.append(np.sum(confidence_eps[eps_iter][:,0]*confidence_eps[eps_iter][:,2])/np.sum(confidence_eps[eps_iter][:,2]))
                x.append(eps[eps_iter])
                y_errorbar.append(np.sum(confidence_eps[eps_iter][:,1]*confidence_eps[eps_iter][:,2])/np.sum(confidence_eps[eps_iter][:,2]))
            ax.errorbar(x,y,yerr=y_errorbar, c='k')
        
        else:
            
            for train_iter in range(len(train_eps)):
                x,y,y_errorbar = [], [], []
                for eps_iter in range(len(eps)):  
                    for file in file_loc[label_iter]:
                        try:
                            confidence_values = np.load(dname+file+'Confidence_Train_Eps_%d_Test_Eps_%d_Seed_11.npy'%(int(train_eps[train_iter]*1000),int(eps[eps_iter]*1000)))
                            predicted_labels = np.load(dname+file+'Labels_Train_Eps_%d_Test_Eps_%d_Seed_11.npy'%(int(train_eps[train_iter]*1000),int(eps[eps_iter]*1000)))
                        except OSError:
                            continue 
                        pred = predicted_labels#torch.argmax(F.softmax(torch.from_numpy(confidence_values),dim=1),dim=1).squeeze()
                        idx = pred==original_labels
                        confidence_values = confidence_values[idx]
                        confidence = torch.max(F.softmax(torch.from_numpy(confidence_values),dim=1),dim=1)
                        confidence_argmax = confidence.indices.squeeze().numpy()
                        confidence = confidence.values.squeeze().numpy()
                        for class_type in range(10):
                            idx = confidence_argmax==class_type
                            subset = confidence[idx]
                        
                            confidence_eps[eps_iter][class_type][0] = np.mean(subset)
                            confidence_eps[eps_iter][class_type][1] = np.std(subset)
                            confidence_eps[eps_iter][class_type][2] = sum(idx)
                
                    y.append(np.sum(confidence_eps[eps_iter][:,0]*confidence_eps[eps_iter][:,2])/np.sum(confidence_eps[eps_iter][:,2]))
                    x.append(eps[eps_iter])
                    y_errorbar.append(np.sum(confidence_eps[eps_iter][:,1]*confidence_eps[eps_iter][:,2])/np.sum(confidence_eps[eps_iter][:,2]))
                ax.errorbar(x,y,yerr=y_errorbar, label=r'VCNN $(\epsilon=%.2f)$'%(train_eps[train_iter]), c=scalarMap.to_rgba(train_eps[train_iter]))
    ax.set_xscale('log')
    ax.set_xlabel(r'Perturbation $\epsilon$',fontsize=25)
    ax.set_ylabel('CONF on Misclassified Adversarial Examples',fontsize=25)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_ylim([0.3,1.05])
    ax.set_xlim([0.05,3.1])
    ax.legend(loc=4,fontsize=15)
    ax.grid()
    [x.set_linewidth(1.5) for x in ax.spines.values()]
def plot_adversarial_accuracy3(fig,ax):
    
    file_loc = [['/results/EP/cel/2023-06-14/17-04-29_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-02_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-08_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-24_gpu0/',
                '/results/EP/cel/2023-06-15/00-48-27_gpu0/'],
                ['/results/BPTT/cel/2023-06-19/17-19-25_gpu0/',
                 '/results/BPTT/cel/2023-06-19/18-28-16_gpu0/',
                 '/results/BPTT/cel/2023-06-19/19-10-16_gpu0/',
                 '/results/BPTT/cel/2023-06-19/19-16-24_gpu0/',
                 '/results/BPTT/cel/2023-06-19/17-17-24_gpu0/'],
                ['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_hardsigmoid.txt'],['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_relu.txt']]
    labels = ['EP',
              'BPTT',
              'Vanilla CNN (HardSigmoid), Loss: CE','Vanilla CNN (ReLU), Loss: CE']
    colors = ['y','m','g','c']
    fname_template = 'Adversarial_Accuracy_Attack_T_%d_Predict_T_%d.txt'
    for label_iter in range(len(labels)):
        if(label_iter>0):
            continue
        param_list = find_files(file_loc[label_iter])
        attack_steps = np.unique(param_list[:,0])
        predict_steps = np.unique(param_list[:,1])
        accuracy_grid = np.zeros((len(attack_steps),len(predict_steps),7,3))
        
        for attack_iter in range(len(attack_steps)):
            for predict_iter in range(len(predict_steps)):
                params = [attack_steps[attack_iter],predict_steps[predict_iter]]
                adversarial_accuracies = []
                for file in file_loc[label_iter]:
                    try:
                        adversarial_accuracy = np.loadtxt(dname+file+fname_template%(params[0],params[1]))
                        adversarial_accuracies.append(adversarial_accuracy)
                    except OSError:
                        continue 
                if(len(adversarial_accuracies)!=0):
                    adversarial_accuracies = np.asarray(adversarial_accuracies)
                    adversarial_accuracies = adversarial_accuracies.reshape((adversarial_accuracies.shape[0]*adversarial_accuracies.shape[1],adversarial_accuracies.shape[-1]))
                    idx = np.argwhere(adversarial_accuracies[:,0]!=0).flatten()
                    adversarial_accuracies = adversarial_accuracies[idx]
                    epsilons = np.unique(adversarial_accuracies[:,0])
                    adversarial_accuracy_stats = np.zeros((len(epsilons),3))
                    for eps_iter in range(len(epsilons)):
                        subset = adversarial_accuracies[adversarial_accuracies[:,0]==epsilons[eps_iter]]
                        adversarial_accuracy_stats[eps_iter] = epsilons[eps_iter], np.mean(subset[:,2]), np.std(subset[:,2])
                    accuracy_grid[attack_iter][predict_iter] = adversarial_accuracy_stats[-7:,]
        epsilons = accuracy_grid[0,0,:,0]
        cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
        cNorm = mcol.Normalize(vmin=min(epsilons), vmax=max(epsilons))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        for predict_iter in range(len(predict_steps)):
            for eps_iter in range(len(epsilons)):
                x = attack_steps
                y = accuracy_grid[:,-1,eps_iter,1]
                x = x[y!=0]
                y = y[y!=0]
                ax.plot(x,y,'.-',c=scalarMap.to_rgba(epsilons[eps_iter]))
        
        ax.set_xlabel(r'Attack Timesteps',fontsize=25)
        ax.set_ylabel('Test Accuracy',fontsize=25)
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax1_divider = make_axes_locatable(ax)
        # Add an Axes to the right of the main Axes.
        cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
        fig.colorbar(scalarMap, cax=cax1)
        cax1.yaxis.set_tick_params(labelsize=15)
        cax1.set_ylabel(r'Perturbation $\epsilon$',fontsize=25)
        ax.grid()
        [x.set_linewidth(1.5) for x in ax.spines.values()]
def plot_adversarial_accuracy_MT(ax):
    
    file_loc = [['/results/EP/cel/2023-05-23/16-52-21_gpu0/Adversarial_Accuracy_UnNormalized.txt',
                '/results/EP/cel/2023-05-23/16-52-21_gpu0/Adversarial_Accuracy_UnNormalized.txt'],
                ['/results/BPTT/cel/2023-05-30/15-31-49_gpu0/Adversarial_Accuracy_UnNormalized.txt'],
                ['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_hardsigmoid.txt'],['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_relu.txt']]
    labels = ['Alg: EP With Symmetric Gradient, Loss: CE',
              'Alg: BPTT, Loss: CE',
              'Vanilla CNN (HardSigmoid), Loss: CE','Vanilla CNN (ReLU), Loss: CE']
    colors = ['k','k','g','c']
    for label_iter in range(len(labels)):
        adversarial_accuracies = []
        for filename in file_loc[label_iter]:
            try:
                adversarial_accuracy = np.loadtxt(dname+filename)
                adversarial_accuracies.append(adversarial_accuracy)
            except OSError:
                continue 
        adversarial_accuracy_mean = np.mean(np.asarray(adversarial_accuracies),axis=0)
        adversarial_accuracy_std = np.std(np.asarray(adversarial_accuracies),axis=0)
        ax.errorbar(adversarial_accuracy_mean[:,0],adversarial_accuracy_mean[:,2], yerr = adversarial_accuracy_std[:,2],label=labels[label_iter],lw=2,c=colors[label_iter])
    ax.set_xlabel(r'Perturbation $\epsilon$',fontsize=25)
    ax.yaxis.tick_right()
    ax.set_ylabel('Test Accuracy',fontsize=25)
    #ax.set_xlim([1,100])
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_ylim([0.05,0.95])
    ax.set_xlim([0.05,3.1])
    ax.legend(loc=3,fontsize=15)
    ax.grid()
    [x.set_linewidth(1.5) for x in ax.spines.values()]
def plot_adversarial_training_accuracy_PlainCNN_MT(ax):
    train_eps = [1]
    cm = plt.get_cmap('jet')
    cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    cNorm = mcol.Normalize(vmin=min(train_eps), vmax=max(train_eps))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    for eps_iter in range(len(train_eps)):
    
        adversarial_accuracies = []
        for seed in [11,13,17,19,31]:
            adversarial_accuracy = np.loadtxt(dname+'/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_Eps_%d_Seed_%d.txt'%(int(train_eps[eps_iter]*1000),seed))
            adversarial_accuracies.append(adversarial_accuracy)
            #ax.plot(adversarial_training[:,0],adversarial_training[:,2],'.-',label=r'VCNN-$\epsilon=%.2f$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
            #ax.plot(np.arange(1,6),adversarial_training[2],'.-',label=r'VCNN ($\epsilon=%.2f)$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
        if(len(adversarial_accuracies)!=0):
            adversarial_accuracy_mean = np.mean(np.asarray(adversarial_accuracies),axis=0)
            adversarial_accuracy_std = np.std(np.asarray(adversarial_accuracies),axis=0)
            ax.errorbar(adversarial_accuracy_mean[:,0],adversarial_accuracy_mean[:,2], yerr = adversarial_accuracy_std[:,2],lw=2,label=r'Adv CNN $(\epsilon=%.2f)$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]))
    ax.legend(loc=3,fontsize=15)
def plot_adversarial_training_accuracy_PlainCNN_Inf_MT(ax):
    train_eps = [0.05,0.5,0.75,1]
    cm = plt.get_cmap('jet')
    cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    cNorm = mcol.Normalize(vmin=min(train_eps), vmax=max(train_eps))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    for eps_iter in range(len(train_eps)):
    
        adversarial_accuracies = []
        for seed in [11,13,17,19,31]:
            adversarial_accuracy = np.loadtxt(dname+'/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Norm_%d_Accuracy_Eps_%d_Seed_%d.txt'%(100,int(train_eps[eps_iter]*1000),seed))
            adversarial_accuracies.append(adversarial_accuracy)
            #ax.plot(adversarial_training[:,0],adversarial_training[:,2],'.-',label=r'VCNN-$\epsilon=%.2f$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
            #ax.plot(np.arange(1,6),adversarial_training[2],'.-',label=r'VCNN ($\epsilon=%.2f)$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
        if(len(adversarial_accuracies)!=0):
            adversarial_accuracy_mean = np.mean(np.asarray(adversarial_accuracies),axis=0)
            adversarial_accuracy_std = np.std(np.asarray(adversarial_accuracies),axis=0)
            ax.errorbar(adversarial_accuracy_mean[:,0],adversarial_accuracy_mean[:,2], yerr = adversarial_accuracy_std[:,2],lw=2,label=r'VCNN $(\epsilon=%.2f)$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]))
    ax.legend(loc=3,fontsize=15)
def prelim_figure():

    fig,ax = plt.subplots(1,2,figsize=(16,8))
    plot_characteristic_time(ax[0])
    plt.tight_layout()
    fig.savefig(dname+'/Characteristic_Time3.png')
def adversarial_example():
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    #plot_characteristic_time(ax[0])
    attack_time = plot_adversarial_accuracy2(ax)
    attack_time = plot_adversarial_accuracy2ll(ax)
    plot_adversarial_accuracyVanilla(ax)
    plot_adversarial_trained_accuracy2ll(ax)
    #plot_adversarial_training_accuracy_PlainCNN(ax)
    plot_adversarial_training_accuracy_PlainCNN_MT(ax)
    plt.tight_layout()
    fig.savefig(dname+'/Adversarial_Accuracy_T_%d.png'%attack_time[0])

def adversarial_example_Inf_Norm():
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    #plot_characteristic_time(ax[0])
    attack_time = plot_adversarial_accuracyinfll(ax)
    plot_adversarial_accuracyVanillaInf(ax)
    #plot_adversarial_training_accuracy_PlainCNN(ax)
    plot_adversarial_training_accuracy_PlainCNN_Inf_MT(ax)
    plt.tight_layout()
    fig.savefig(dname+'/Adversarial_Accuracy_Inf.png')
def adversarial_timesteps():
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    #plot_characteristic_time(ax[0])
    plot_adversarial_accuracy3(fig,ax)
    
    plt.tight_layout()
    fig.savefig(dname+'/Adversarial_Accuracy_Timesteps.png')

def adversarial_confidence():
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    #plot_characteristic_time(ax[0])
    plot_confidence(ax)
    plt.tight_layout()
    fig.savefig(dname+'/Confidence.png')

#adversarial_timesteps()
adversarial_example()
#adversarial_example_Inf_Norm()
#adversarial_confidence()