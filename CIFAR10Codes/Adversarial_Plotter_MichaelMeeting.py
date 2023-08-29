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
def find_filesll(file_loc_list,attack_norm=-1):
    param_list = []
    for i in range(len(file_loc_list)):
        if attack_norm != -1:
            for filename in glob.glob(dname+file_loc_list[i]+'White_Last_Layer_Norm_%d_Adversarial_Accuracy_A*.txt'%attack_norm):
                fname = filename.replace(dname+file_loc_list[i],'')
                param_list.append(re.findall(r'\d+',fname))
        else:
            for filename in glob.glob(dname+file_loc_list[i]+'White_Last_Layer_Adversarial_Accuracy_A*.txt'):
                fname = filename.replace(dname+file_loc_list[i],'')
                param_list.append(re.findall(r'\d+',fname))
    param_list = np.asarray(param_list,dtype=int)
    param_list = np.unique(param_list,axis=0)
    if attack_norm !=-1:
        return param_list[:,[1,2]]
    else:
        return param_list
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
        ax.set_xlim([0.005,3])
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
                         adversarial_accuracy_stats = np.vstack(([0.005,adversarial_accuracy_stats[0,1],0],adversarial_accuracy_stats))
                         ax.plot(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],'.-',
                                    label='%s $T_{attack}=%d$'%(labels[label_iter],params[0]), lw=2,c=scalarMap.to_rgba(params[0]))
                    else:
                        adversarial_accuracy_stats = np.vstack(([0.005,adversarial_accuracy_stats[0,1],0],adversarial_accuracy_stats))
                        ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1], yerr = adversarial_accuracy_stats[:,2],label='%s $T_{attack}=%d$'%(labels[label_iter],params[0]),lw=2,c=colors[label_iter])
    #if(len(adversarial_accuracies)!=0):
    ax.set_xlabel(r'$l_2$ Perturbation $\epsilon$',fontsize=30)
    ax.set_ylabel('Test Accuracy',fontsize=30)
    #ax.set_xlim([1,100])
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.set_ylim([0.05,0.95])
    ax.set_xlim([0.005,3.1])
    ax.legend(loc=3,fontsize=30)
    ax.grid()
    [x.set_linewidth(1.5) for x in ax.spines.values()]
    return 0#np.unique(param_list[:,0])
def plot_adversarial_accuracyVanilla(ax):
    
    file_loc = [['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Norm_2_Accuracy_Model_hardsigmoid_Seed_1.txt'],['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Norm_2_Accuracy_Eps_0_Seed_%d.txt']]
    labels = ['VCNN (HardSigmoid)','CNN']
    colors = ['g','c']
    for label_iter in range(1,len(labels)):
        adversarial_accuracies = []
        for seed in [11,13,17,19,31]:
            try:
                adversarial_accuracy = np.loadtxt(dname+file_loc[label_iter][0]%(seed))
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
            adversarial_accuracy_stats = np.vstack(([0.005,adversarial_accuracy_stats[0,1],0],adversarial_accuracy_stats))
            ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],yerr =adversarial_accuracy_stats[:,2],
                                label=labels[label_iter], color='y',lw=4)
def plot_adversarial_accuracyVanillaInf(ax):
    
    file_loc = [['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Norm_%d_Accuracy_Model_hardsigmoid_Seed_11.txt'],['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Norm_%d_Accuracy_Eps_0_Seed_%d.txt']]
    labels = ['VCNN (HardSigmoid)','CNN']
    colors = ['g','c']
    for label_iter in range(1,len(labels)):
        adversarial_accuracies = []
        for seed in [11,13,17,19,31]:
            try:
                adversarial_accuracy = np.loadtxt(dname+file_loc[label_iter][0]%(100,seed))
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
            #adversarial_accuracy_stats = np.vstack(([0.005,adversarial_accuracy_stats[0,1],0],adversarial_accuracy_stats))
            print(adversarial_accuracy_stats[:,2])
            ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],yerr = adversarial_accuracy_stats[:,2],
                                label=labels[label_iter],c='y', lw=4)
def plot_adversarial_accuracyVanillaHSJ(ax,norm=2):
    max_evals = 1000
    file_loc = [['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_HSJ_Norm_%d_Accuracy_Model_hardsigmoid_Max_Evals_%d_Seed_11.txt'],['/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_HSJ_Norm_%d_Accuracy_Eps_0_Max_Evals_%d_Seed_%d.txt']]
    labels = ['CNN (HardSigmoid)','CNN']
    colors = ['g','c']
    for label_iter in range(1,len(labels)):
        adversarial_accuracies = []
        for seed in [11,13,17,19,31]:
            try:
                adversarial_accuracy = np.loadtxt(dname+ file_loc[label_iter][0]%(norm,max_evals*1000,seed))
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
            #adversarial_accuracy_stats = np.vstack(([0.005,adversarial_accuracy_stats[0,1],0],adversarial_accuracy_stats))
            ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],yerr = adversarial_accuracy_stats[:,2],
                                label=labels[label_iter],color='y', lw=4)                
def plot_adversarial_accuracy2ll(ax,T=-1):
    
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
    fname_template2 = 'White_Last_Layer_Norm_2_Adversarial_Accuracy_Attack_T_%d_Predict_T_%d.txt'
    for label_iter in range(len(labels)):
        if(label_iter>0):
            continue
        param_list = find_filesll(file_loc[label_iter])
        if(T!=-1):
            param_list = param_list[param_list[:,0]==T]

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
                    print(file)
                    try:

                        adversarial_accuracy = np.loadtxt(dname+file+fname_template2%(params[0],params[1]))
                        adversarial_accuracies.append(adversarial_accuracy)
                    except OSError:
                        print(file)
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
                        adversarial_accuracy_stats = np.vstack(([0.005,0.88,0],adversarial_accuracy_stats))
                        if T!=-1:
                            ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],yerr = adversarial_accuracy_stats[:,2],
                                        label='%s CNN'%(labels[label_iter]), lw=4,c=scalarMap.to_rgba(params[0]))
                        else:
                            print(param_list[0])
                            ax.plot(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],'.-',
                                        label='%s T=%d'%(labels[label_iter],params[0]), lw=4,c=scalarMap.to_rgba(params[0]))
                    else:
                        adversarial_accuracy_stats = np.vstack(([0.005,0.88,0],adversarial_accuracy_stats))
                        ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1], yerr = adversarial_accuracy_stats[:,2],label='%s $T_{attack}=%d$'%(labels[label_iter],params[0]),lw=4,c=colors[label_iter])
    return np.unique(param_list[:,0])
def plot_adversarial_accuracyinfll(ax,T=29):
    
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
        param_list = find_filesll(file_loc[label_iter],attack_norm=100)
        
        param_list = param_list[param_list[:,0]==T]

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
                    print(file)
                    continue 
            
            if(len(adversarial_accuracies)!=0):
                adversarial_accuracies = np.asarray(adversarial_accuracies)
                print(adversarial_accuracies.shape)
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
                                    label='%s '%(labels[label_iter]), lw=4,c='k')
                    else:
                        ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1], yerr = adversarial_accuracy_stats[:,2],label='%s'%(labels[label_iter]),lw=2,c=colors[label_iter])
    
    ax.set_xlabel(r'$l_\infty$ Perturbation $\epsilon$',fontsize=30)
    ax.set_ylabel('Test Accuracy',fontsize=30)
    #ax.set_xlim([1,100])
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.set_ylim([0.05,0.95])
    ax.set_xlim([0.0001,0.11])
    ax.legend(loc=3,fontsize=15)
    ax.grid()
    [x.set_linewidth(1.5) for x in ax.spines.values()]
    return np.unique(param_list[:,0])
def plot_adversarial_accuracydifftrain(ax):
    
    file_loc = [['/results/EP/cel/2023-07-07/10-56-31_gpu0/',
                '/results/EP/cel/2023-07-09/12-26-07_gpu0/',
                '/results/EP/cel/2023-07-10/10-46-00_gpu0/']]
    labels = ['EP']
    colors = ['y','m','g','c']
    fname_template = 'White_Last_Layer_Norm_%d_Adversarial_Accuracy_Attack_T_%d_Predict_T_%d.txt'
    for label_iter in range(len(labels)):
        
        param_list = find_filesll(file_loc[label_iter],attack_norm=2)
        print(param_list)
        
        cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["k","c"])
        cNorm = mcol.LogNorm(vmin=min(param_list[:,0]), vmax=max(param_list[:,0]))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        for params in param_list:
            
            print(labels[label_iter],params)
            adversarial_accuracies = []
            for file in file_loc[label_iter]:
                try:
                    ##print(dname+file+fname_template%(params[0],params[1]))
                    adversarial_accuracy = np.loadtxt(dname+file+fname_template%(2,params[0],params[1]))
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
                        adversarial_accuracy_stats = np.vstack(([0.005,adversarial_accuracy_stats[0,1],0],adversarial_accuracy_stats))
                        ax.plot(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],'--',
                                    label='%s $T_{attack}=%d$'%(labels[label_iter],params[0]), lw=2,c=scalarMap.to_rgba(params[0]))
                    else:
                        adversarial_accuracy_stats = np.vstack(([0.005,adversarial_accuracy_stats[0,1],0],adversarial_accuracy_stats))
                        ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1], yerr = adversarial_accuracy_stats[:,2],label='%s $T_{attack}=%d$'%(labels[label_iter],params[1]),lw=2,c=colors[label_iter])
    
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
    ax.set_xlim([0.005,3.1])
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
    ax.set_xlim([0.005,3.1])
    ax.legend(loc=3,fontsize=15)
    ax.grid()
    [x.set_linewidth(1.5) for x in ax.spines.values()]
def plot_adversarial_accuracy_ViT(ax,norm=2):
    file_loc = '/ViT-CIFAR/Norm_%d_Adversarial_Accuracy_Patch_%d_Augment_%d_Seed_%d.txt'
    patches = [4,8]
    augments = [1]
    adversarial_accuracies = []
    for patch in [4]:
        for augment in augments:
            for seed in [11,13,17,19,31]:
                try:
                    data = np.loadtxt(dname+file_loc%(norm,patch,augment,seed))
                except FileNotFoundError:
                    continue
                adversarial_accuracies.append(data)
            
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
        if(norm==2):
            adversarial_accuracy_stats = np.vstack(([0.005,adversarial_accuracies[-1,1],0],adversarial_accuracy_stats))
        else:
            adversarial_accuracy_stats = np.vstack(([0.0001,adversarial_accuracies[-1,1],0],adversarial_accuracy_stats))
        ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1], yerr = adversarial_accuracy_stats[:,2],label='ViT',lw=4,c='c')
    return 
def plot_adversarial_accuracy_ViT_HSJ(ax,norm=2):
    file_loc = '/ViT-CIFAR/HSJ_Norm_%d_Adversarial_Accuracy_Patch_%d_Augment_%d_Seed_%d.txt'
    patches = [4,8]
    augments = [1]
    adversarial_accuracies = []
    for patch in [4]:
        for augment in augments:
            for seed in [11,13,17,19,31]:
                try:
                    data = np.loadtxt(dname+file_loc%(norm,patch,augment,seed))
                except FileNotFoundError:
                    continue
                adversarial_accuracies.append(data)
            
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
        if(norm==2):
            adversarial_accuracy_stats = np.vstack(([0.005,adversarial_accuracies[-1,1],0],adversarial_accuracy_stats))
        else:
            adversarial_accuracy_stats = np.vstack(([0.0001,adversarial_accuracies[-1,1],0],adversarial_accuracy_stats))
        ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1], yerr = adversarial_accuracy_stats[:,2],label='ViT',lw=4,c='c')
    return
def plot_adversarial_training_accuracy_PlainCNN_MT(ax,norm=2):
    train_eps = [0.5,1]
    cm = plt.get_cmap('jet')
    cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    cNorm = mcol.Normalize(vmin=min(train_eps), vmax=max(train_eps))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    for eps_iter in range(len(train_eps)):
    
        adversarial_accuracies = []
        for seed in [13,17,19,31]:
            if norm == 2:
                adversarial_accuracy = np.loadtxt(dname+'/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_Eps_%d_Seed_%d.txt'%(int(train_eps[eps_iter]*1000),seed))
            else:
                adversarial_accuracy = np.loadtxt(dname+'/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Norm_%d_Accuracy_Eps_%d_Seed_%d.txt'%(norm,int(train_eps[eps_iter]*1000),seed))
            adversarial_accuracies.append(adversarial_accuracy)
            print(seed,adversarial_accuracy.shape)
            #ax.plot(adversarial_training[:,0],adversarial_training[:,2],'.-',label=r'VCNN-$\epsilon=%.2f$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
            #ax.plot(np.arange(1,6),adversarial_training[2],'.-',label=r'VCNN ($\epsilon=%.2f)$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
        #print(np.asarray(adversarial_accuracies))
        if(len(adversarial_accuracies)!=0):
            adversarial_accuracy_mean = np.mean(np.asarray(adversarial_accuracies),axis=0)
            adversarial_accuracy_std = np.std(np.asarray(adversarial_accuracies),axis=0)
            if norm ==2:
                adversarial_accuracy_mean = np.vstack(([0.005,adversarial_accuracy_mean[0,2],adversarial_accuracy_mean[0,2]],adversarial_accuracy_mean))
                adversarial_accuracy_std = np.vstack(([0.005,0,0],adversarial_accuracy_std))
            else:
                adversarial_accuracy_mean = np.vstack(([0.0005,adversarial_accuracy_mean[0,2],adversarial_accuracy_mean[0,2]],adversarial_accuracy_mean))
                adversarial_accuracy_std = np.vstack(([0.0005,0,0],adversarial_accuracy_std))
            ax.errorbar(adversarial_accuracy_mean[:,0],adversarial_accuracy_mean[:,2], yerr = adversarial_accuracy_std[:,2],lw=4,label=r'CNN $(\epsilon=%.2f)$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]))
            
    ax.legend(loc=3,fontsize=30)
def plot_adversarial_training_accuracy_PlainCNN_MT_HSJ(ax,norm=2):
    train_eps = [0.05,0.5,0.75,1]
    cm = plt.get_cmap('jet')
    cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    cNorm = mcol.Normalize(vmin=min(train_eps), vmax=max(train_eps))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    max_evals = 1000
    for eps_iter in range(len(train_eps)):
    
        adversarial_accuracies = []
        for seed in [11,13,17,19,31]:
            adversarial_accuracy = np.loadtxt(dname+'/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_HSJ_Norm_%d_Accuracy_Eps_%d_Max_Evals_%d_Seed_%d.txt'%(norm,int(train_eps[eps_iter]*1000),int(max_evals*1000),seed))
            adversarial_accuracies.append(adversarial_accuracy)
            #ax.plot(adversarial_training[:,0],adversarial_training[:,2],'.-',label=r'VCNN-$\epsilon=%.2f$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
            #ax.plot(np.arange(1,6),adversarial_training[2],'.-',label=r'VCNN ($\epsilon=%.2f)$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
        if(len(adversarial_accuracies)!=0):
            adversarial_accuracy_mean = np.mean(np.asarray(adversarial_accuracies),axis=0)
            adversarial_accuracy_std = np.std(np.asarray(adversarial_accuracies),axis=0)
            #adversarial_accuracy_mean = np.vstack(([0.005,adversarial_accuracy_mean[0,2],adversarial_accuracy_mean[0,2]],adversarial_accuracy_mean))
            #adversarial_accuracy_std = np.vstack(([0.005,0,0],adversarial_accuracy_std))
            adversarial_accuracy_std = adversarial_accuracy_std[adversarial_accuracy_mean[:,0]!=0]
            
            adversarial_accuracy_mean = adversarial_accuracy_mean[adversarial_accuracy_mean[:,0]!=0]
            ax.errorbar(adversarial_accuracy_mean[:,0],adversarial_accuracy_mean[:,2], yerr = adversarial_accuracy_std[:,2],lw=4,label=r'CNN $(\epsilon=%.2f)$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]))
    
    if norm==100:
        ax.set_xlabel(r'$l_\infty$ Perturbation $\epsilon$',fontsize=30)
    else:
        ax.set_xlabel(r'$l_2$ Perturbation $\epsilon$',fontsize=30)
    ax.set_ylabel('Test Accuracy',fontsize=30)
    ax.set_xlim([0.0007,0.047])
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.set_ylim([0.05,0.95])
    #ax.set_title('Max Evals = %d'%(3*max_evals),fontsize=25)
    #ax.set_xlim([0.0001,0.11])
    ax.legend(loc=3,fontsize=30)
    ax.grid()
    [x.set_linewidth(1.5) for x in ax.spines.values()]
    
def plot_adversarial_accuracy_HSJ(ax,norm=2):
    
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
    labels = ['EP CNN',
              'BPTT',
              'VCNN (HardSigmoid)','VCNN (ReLU)']
    colors = ['k','m','g','c']
    if(norm==100):
        fname_template = 'HSJ_Norm_%d_Adversarial_Accuracy_Max_Evals_%d.txt'
    else:
        fname_template = 'HSJ_Adversarial_Accuracy_Max_Evals_%d.txt'
    for label_iter in range(len(labels)):
        
        if(label_iter<1):
          
            
            adversarial_accuracies = []
            for file in file_loc[label_iter]:
                try:
                    ##print(dname+file+fname_template%(params[0],params[1]))
                    #print(dname+file+fname_template%(1000*1000))
                    if norm ==100:
                        adversarial_accuracy = np.loadtxt(dname+file+fname_template%(100,1000*1000))
                    else:
                        adversarial_accuracy = np.loadtxt(dname+file+fname_template%(1000*1000))
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
                ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],yerr = adversarial_accuracy_stats[:,2],
                                    label='%s'%(labels[label_iter]), lw=4,c='k')
    ax.legend(loc=3,fontsize=30)              
    return 
def plot_adversarial_accuracy_HSJ_CIFAR100(ax,norm=2):    
    file_loc = [['/CIFAR100/results/EP/cel/2023-07-26/09-20-56_gpu0/',
                 '/CIFAR100/results/EP/cel/2023-08-11/11-31-07_gpu0/',
                 '/CIFAR100/results/EP/cel/2023-08-11/12-04-36_gpu0/',
                 '/CIFAR100/results/EP/cel/2023-08-11/12-11-38_gpu0/',
                 '/CIFAR100/results/EP/cel/2023-08-11/21-31-29_gpu0/'],
                ['/CIFAR100/results/MTModel/']]
    labels = ['EP CNN',
              'CNN']
    colors = ['k','y','g','c']
    max_evals = 1000
    fname_template = ['HSJ_Norm_%d_Adversarial_Accuracy_Max_Evals_%d.txt',
                      'Adversarial_HSJ_Norm_%d_Accuracy_Model_relu_Max_Evals_%d_Seed_%d.txt']
    print(dname+file_loc[0][0]+fname_template[0]%(100,1000*1000))
    for label_iter in range(len(labels)):
        adversarial_accuracies = []
        for file in file_loc[label_iter]:
            try: 
                if label_iter ==0:
                    adversarial_accuracy = np.loadtxt(dname+file+fname_template[label_iter]%(norm,1000*1000))
                    print(dname+file+fname_template[label_iter]%(norm,1000*1000))
                    adversarial_accuracies.append(adversarial_accuracy)
                else:
                    for seed in [11,13,17,19,31]:
                        print(dname+file+fname_template[label_iter]%(norm,1000*1000,seed))
                        adversarial_accuracy = np.loadtxt(dname+file+fname_template[label_iter]%(norm,1000*1000,seed))
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
            ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],yerr = adversarial_accuracy_stats[:,2],
                                label='%s'%(labels[label_iter]), lw=4,c=colors[label_iter])
    ax.legend(loc=3,fontsize=30)              
    return
def plot_adversarial_training_accuracy_CIFAR100_HSJ(ax,norm=2):
    train_eps = [0.05,0.5,0.75,1]
    cm = plt.get_cmap('jet')
    cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    cNorm = mcol.Normalize(vmin=min(train_eps), vmax=max(train_eps))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    max_evals = 1000
    for eps_iter in range(len(train_eps)):
    
        adversarial_accuracies = []
        for seed in [13,17,19,31]:
            adversarial_accuracy = np.loadtxt(dname+'/CIFAR100/results/MTModel/Adversarial_HSJ_Norm_%d_Accuracy_Eps_%d_Max_Evals_%d_Seed_%d.txt'%(norm,int(train_eps[eps_iter]*1000),int(max_evals*1000),seed))
            adversarial_accuracies.append(adversarial_accuracy)
            #ax.plot(adversarial_training[:,0],adversarial_training[:,2],'.-',label=r'VCNN-$\epsilon=%.2f$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
            #ax.plot(np.arange(1,6),adversarial_training[2],'.-',label=r'VCNN ($\epsilon=%.2f)$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
        if(len(adversarial_accuracies)!=0):
            adversarial_accuracy_mean = np.mean(np.asarray(adversarial_accuracies),axis=0)
            adversarial_accuracy_std = np.std(np.asarray(adversarial_accuracies),axis=0)
            #adversarial_accuracy_mean = np.vstack(([0.005,adversarial_accuracy_mean[0,2],adversarial_accuracy_mean[0,2]],adversarial_accuracy_mean))
            #adversarial_accuracy_std = np.vstack(([0.005,0,0],adversarial_accuracy_std))
            adversarial_accuracy_std = adversarial_accuracy_std[adversarial_accuracy_mean[:,0]!=0]
            
            adversarial_accuracy_mean = adversarial_accuracy_mean[adversarial_accuracy_mean[:,0]!=0]
            ax.errorbar(adversarial_accuracy_mean[:,0],adversarial_accuracy_mean[:,2], yerr = adversarial_accuracy_std[:,2],lw=4,label=r'CNN $(\epsilon=%.2f)$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]))
def plot_data(adversarial_accuracies,ax,label,color):
    if(len(adversarial_accuracies)!=0):
        print(len(adversarial_accuracies))
        adversarial_accuracies = np.asarray(adversarial_accuracies)
        adversarial_accuracies = adversarial_accuracies.reshape((adversarial_accuracies.shape[0]*adversarial_accuracies.shape[1],adversarial_accuracies.shape[-1]))
        idx = np.argwhere(adversarial_accuracies[:,0]!=0).flatten()
        adversarial_accuracies = adversarial_accuracies[idx]
        epsilons = np.unique(adversarial_accuracies[:,0])
        adversarial_accuracy_stats = np.zeros((len(epsilons),3))
        
        for eps_iter in range(len(epsilons)):
            subset = adversarial_accuracies[adversarial_accuracies[:,0]==epsilons[eps_iter]]
            adversarial_accuracy_stats[eps_iter] = epsilons[eps_iter], np.mean(subset[:,2]), np.std(subset[:,2])
        ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],yerr = adversarial_accuracy_stats[:,2],
                            label='%s'%(label), lw=4,c=color)
def plot_AA_CIFAR10(ax,norm=100):    
    file_loc = [['/results/EP/cel/2023-06-14/17-04-29_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-02_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-08_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-24_gpu0/',
                '/results/EP/cel/2023-06-15/00-48-27_gpu0/'],
                ['/results/BaselineCNN/cel/Jupyter/MTModel/'],['/ViT-CIFAR/']]
    labels = ['EP CNN',
              'CNN','ViT']
    colors = ['k','y','c']
    max_evals = 1000
    cm = plt.get_cmap('jet')
    cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    cNorm = mcol.Normalize(vmin=0.05, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fname_template = ['AA_Norm_%d_Adversarial_Accuracy_Max_Evals_%d.txt',
                      'AA_Adversarial_Norm_%d_Accuracy_Eps_%d_Seed_%d.txt',
                      'AA_Norm_%d_Adversarial_Accuracy_Patch_4_Augment_1_Seed_%d.txt']
    print(dname+file_loc[0][0]+fname_template[0]%(100,1000*1000))
    for label_iter in range(len(labels)):
        
        try: 
            if label_iter ==0:
                adversarial_accuracies = []
                for file in file_loc[label_iter]:
                    adversarial_accuracy = np.loadtxt(dname+file+fname_template[label_iter]%(norm,1000*1000))
                    adversarial_accuracies.append(adversarial_accuracy)
                plot_data(adversarial_accuracies,ax,labels[label_iter],colors[label_iter])
            if label_iter == 1:
                file = file_loc[label_iter][0]
                for train_eps in [0,0.05,0.5,0.75,1]:
                    adversarial_accuracies = []
                    for seed in [11,13,17,19,31]:
                        adversarial_accuracy = np.loadtxt(dname+file+fname_template[label_iter]%(norm,int(train_eps*1000),seed))
                        adversarial_accuracies.append(adversarial_accuracy)
                    if train_eps == 0:
                        plot_data(adversarial_accuracies,ax,labels[label_iter],colors[label_iter])
                    else:
                        plot_data(adversarial_accuracies,ax,
                                    labels[label_iter]+r' $(\epsilon=%.2f)$'%train_eps,scalarMap.to_rgba(train_eps))
            if label_iter == 2:
                adversarial_accuracies = []
                file = file_loc[label_iter][0]
                for seed in [11,13,17,19,31]:
                    adversarial_accuracy = np.loadtxt(dname+file+fname_template[label_iter]%(norm,seed))
                    adversarial_accuracies.append(adversarial_accuracy)
                plot_data(adversarial_accuracies,ax,labels[label_iter],colors[label_iter])
        except OSError:
            continue 
       
            
    return
def plot_adversarial_accuracy_CIFAR100(ax,norm=2):    
    file_loc = [['/CIFAR100/results/EP/cel/2023-07-26/09-20-56_gpu0/',
                 '/CIFAR100/results/EP/cel/2023-08-11/11-31-07_gpu0/',
                 '/CIFAR100/results/EP/cel/2023-08-11/12-04-36_gpu0/',
                 '/CIFAR100/results/EP/cel/2023-08-11/12-11-38_gpu0/',
                 '/CIFAR100/results/EP/cel/2023-08-11/21-31-29_gpu0/'],
                ['/CIFAR100/results/MTModel/']]
    labels = ['EP CNN',
              'CNN']
    colors = ['k','y','g','c']
    max_evals = 1000
    fname_template = ['White_Last_Layer_Norm_%d_Adversarial_Accuracy_Attack_T_70_Predict_T_250.txt',
                      'Adversarial_Norm_%d_Accuracy_Model_relu_Max_Evals_%d_Seed_%d.txt']
    print(dname+file_loc[0][0]+fname_template[0]%(100))
    for label_iter in range(1):
        adversarial_accuracies = []
        for file in file_loc[label_iter]:
            try: 
                if label_iter ==0:
                    adversarial_accuracy = np.loadtxt(dname+file+fname_template[label_iter]%(norm))
                    if len(adversarial_accuracy.shape)==1:
                        continue
                    print(dname+file+fname_template[label_iter]%(norm),adversarial_accuracy.shape)
                    
                    adversarial_accuracies.append(adversarial_accuracy)
                else:
                    for seed in [11,13,17,19,31]:
                        print(dname+file+fname_template[label_iter]%(norm,1000*1000,seed))
                        adversarial_accuracy = np.loadtxt(dname+file+fname_template[label_iter]%(norm,1000*1000,seed))
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
            ax.errorbar(adversarial_accuracy_stats[:,0],adversarial_accuracy_stats[:,1],yerr = adversarial_accuracy_stats[:,2],
                                label='%s'%(labels[label_iter]), lw=4,c=colors[label_iter])
    ax.legend(loc=3,fontsize=30)              
    return
def plot_adversarial_training_accuracy_CIFAR100(ax,norm=2):
    train_eps = [0.05,0.5,0.75,1]
    cm = plt.get_cmap('jet')
    cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    cNorm = mcol.Normalize(vmin=min(train_eps), vmax=max(train_eps))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    max_evals = 1000
    for eps_iter in range(len(train_eps)):
    
        adversarial_accuracies = []
        for seed in [11,13,17,19,31]:
            adversarial_accuracy = np.loadtxt(dname+'/CIFAR100/results/MTModel/Adversarial_Norm_%d_Accuracy_Eps_%d_Seed_%d.txt'%(norm,int(train_eps[eps_iter]*1000),seed))
            adversarial_accuracies.append(adversarial_accuracy)
            #ax.plot(adversarial_training[:,0],adversarial_training[:,2],'.-',label=r'VCNN-$\epsilon=%.2f$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
            #ax.plot(np.arange(1,6),adversarial_training[2],'.-',label=r'VCNN ($\epsilon=%.2f)$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
        print(np.asarray(adversarial_accuracies).shape)
        if(len(adversarial_accuracies)!=0):
            adversarial_accuracy_mean = np.mean(np.asarray(adversarial_accuracies),axis=0)
            adversarial_accuracy_std = np.std(np.asarray(adversarial_accuracies),axis=0)
            #adversarial_accuracy_mean = np.vstack(([0.005,adversarial_accuracy_mean[0,2],adversarial_accuracy_mean[0,2]],adversarial_accuracy_mean))
            #adversarial_accuracy_std = np.vstack(([0.005,0,0],adversarial_accuracy_std))
            adversarial_accuracy_std = adversarial_accuracy_std[adversarial_accuracy_mean[:,0]!=0]
            
            adversarial_accuracy_mean = adversarial_accuracy_mean[adversarial_accuracy_mean[:,0]!=0]
            ax.errorbar(adversarial_accuracy_mean[:,0],adversarial_accuracy_mean[:,2], yerr = adversarial_accuracy_std[:,2],lw=4,label=r'CNN $(\epsilon=%.2f)$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]))

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
    #plot_adversarial_training_accuracy_PlainCNN(ax)
    plot_adversarial_training_accuracy_PlainCNN_MT(ax)
    plt.tight_layout()
    fig.savefig(dname+'/Adversarial_Accuracy_T_%d.png'%attack_time[0])

def adversarial_example_Inf_Norm():
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    #plot_characteristic_time(ax[0])
    attack_time = plot_adversarial_accuracyinfll(ax,70)
    plot_adversarial_accuracyVanillaInf(ax)
    plot_adversarial_accuracy_ViT(ax,norm=100)
    #plot_adversarial_training_accuracy_PlainCNN(ax)
    plot_adversarial_training_accuracy_PlainCNN_MT(ax,norm=100)
    plt.tight_layout()
    fig.savefig(dname+'/Adversarial_Accuracy_Inf.png')
    fig.savefig(dname+'/Adversarial_Accuracy_Inf.pdf')
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
def adversarial_example_EP():
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    #plot_characteristic_time(ax[0])
    attack_time = plot_adversarial_accuracy2(ax)
    attack_time = plot_adversarial_accuracy2ll(ax)
    ax.set_xlabel(r'$l_2$ Perturbation $\epsilon$',fontsize=30)
    ax.set_ylabel('Test Accuracy',fontsize=30)
    #ax.set_xlim([1,100])
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.set_ylim([0.05,0.95])
    ax.set_xlim([0.005,3.1])
    ax.legend(loc=3,fontsize=30)
    #ax.grid()
    [x.set_linewidth(1.5) for x in ax.spines.values()]
    plt.tight_layout()
    fig.savefig(dname+'/Adversarial_Accuracy_Timesteps.pdf')
def adversarial_example_EPOptimized():
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    #plot_characteristic_time(ax[0])
    attack_time = plot_adversarial_accuracy2(ax)
    attack_time = plot_adversarial_accuracy2ll(ax,T=70)
    attack_time = plot_adversarial_accuracydifftrain(ax)
    #plot_adversarial_training_accuracy_PlainCNN_MT(ax)
    ax.set_xlabel(r'$l_2$ Perturbation $\epsilon$',fontsize=25)
    ax.set_ylabel('Test Accuracy',fontsize=25)
    #ax.set_xlim([1,100])
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_ylim([0.05,0.95])
    ax.set_xlim([0.005,3.1])
    ax.legend(loc=3,fontsize=15)
    ax.grid()
    [x.set_linewidth(1.5) for x in ax.spines.values()]
    plt.tight_layout()
    fig.savefig(dname+'/Adversarial_Accuracy_Optimal.png')
def adversarial_example_CNN():
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    #plot_characteristic_time(ax[0])
    attack_time = plot_adversarial_accuracy2(ax)
    attack_time = plot_adversarial_accuracy2ll(ax,70)
    plot_adversarial_accuracy_ViT(ax,norm=2)
    plot_adversarial_accuracyVanilla(ax)
    ax.legend(loc=3,fontsize=15)
    #plot_adversarial_training_accuracy_PlainCNN(ax)
    #plot_adversarial_training_accuracy_PlainCNN_MT(ax)
    plt.tight_layout()
    fig.savefig(dname+'/Adversarial_Accuracy_CNN_M.pdf')
def adversarial_example_CNN_Adv():
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    #plot_characteristic_time(ax[0])
    attack_time = plot_adversarial_accuracy2(ax)
    attack_time = plot_adversarial_accuracy2ll(ax,70)
    plot_adversarial_accuracy_ViT(ax,norm=2)
    plot_adversarial_accuracyVanilla(ax)
    ax.legend(loc=3,fontsize=30)
    #plot_adversarial_training_accuracy_PlainCNN(ax)
    plot_adversarial_training_accuracy_PlainCNN_MT(ax)
    plt.tight_layout()
    fig.savefig(dname+'/Adversarial_Accuracy_CNN_Adv_M.pdf')
def characteristic_time():
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    file_loc = ['/results/EP/cel/2023-06-14/17-04-29_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-02_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-08_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-24_gpu0/',
                '/results/EP/cel/2023-06-15/00-48-27_gpu0/']
    characteristic_time = []
    for file in file_loc:
        data = np.loadtxt(dname+file+'Characteristic_Time.txt')
        characteristic_time.append(data)
    characteristic_time = np.asarray(characteristic_time)
    #characteristic_time = characteristic_time.reshape(characteristic_time.shape[0]*characteristic_time.shape[1],characteristic_time.shape[-1])
    time_mean = np.mean(characteristic_time,axis=0)
    time_std = np.std(characteristic_time, axis=0)
    ax.errorbar(time_mean[:,0],time_mean[:,1],yerr=time_std[:,1],lw=4,c='k')
    ax.set_xscale('log')
    ax.set_xlabel(r'Free Phase Iteration \#',fontsize=30)
    ax.set_ylabel('Test Accuracy',fontsize=30)
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.set_ylim([0.05,0.95])
    ax.set_xlim([1,100])
    plt.tight_layout()
    fig.savefig('Characteristic_Time.pdf')
def adversarial_example_combined():
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    #plot_characteristic_time(ax[0])
    attack_time = plot_adversarial_accuracy2(ax)
    attack_time = plot_adversarial_accuracy2ll(ax,10)
    plot_adversarial_accuracyVanilla(ax)
    #plot_adversarial_training_accuracy_PlainCNN(ax)
    plot_adversarial_training_accuracy_PlainCNN_MT(ax)
    plt.tight_layout()
    fig.savefig(dname+'/Adversarial_Accuracy_Combined.png')
def adversarial_HSJ():
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    #plot_characteristic_time(ax[0])
    norm=100
    plot_adversarial_accuracy_HSJ(ax,norm)
    plot_adversarial_accuracy_ViT_HSJ(ax,norm)
    plot_adversarial_accuracyVanillaHSJ(ax,norm)
    plot_adversarial_training_accuracy_PlainCNN_MT_HSJ(ax,norm)
    plt.tight_layout()
    fig.savefig(dname+'/Square_Attack_Accuracy_%d.pdf'%norm)
    fig.savefig(dname+'/Square_Attack_Accuracy_%d.png'%norm)
def CIFAR100_HSJ():
    norm = 100
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    plot_adversarial_accuracy_HSJ_CIFAR100(ax,norm)
    plot_adversarial_training_accuracy_CIFAR100_HSJ(ax,norm)
    if norm==100:
        ax.set_xlabel(r'$l_\infty$ Perturbation $\epsilon$',fontsize=30)
    else:
        ax.set_xlabel(r'$l_2$ Perturbation $\epsilon$',fontsize=30)
    ax.set_ylabel('Test Accuracy',fontsize=30)
    ax.set_xlim([0.0007,0.047])
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.set_ylim([0.05,0.71])
    #ax.set_title('Max Evals = %d'%(3*max_evals),fontsize=25)
    #ax.set_xlim([0.0001,0.11])
    ax.legend(loc=3,fontsize=30)
    ax.grid()
    [x.set_linewidth(1.5) for x in ax.spines.values()]
    plt.tight_layout()
    fig.savefig(dname+'/CIFAR100_Square_Attack_Accuracy_%d.pdf'%norm)
    fig.savefig(dname+'/CIFAR100_Square_Attack_Accuracy_%d.png'%norm)
def CIFAR10_AA():
    norm = 100
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    plot_AA_CIFAR10(ax,norm)
    if norm==100:
        ax.set_xlabel(r'$l_\infty$ Perturbation $\epsilon$',fontsize=30)
    else:
        ax.set_xlabel(r'$l_2$ Perturbation $\epsilon$',fontsize=30)
    ax.set_ylabel('Test Accuracy',fontsize=30)
    ax.set_xlim([0.0007,0.047])
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.set_ylim([0.05,1])
    #ax.set_title('Max Evals = %d'%(3*max_evals),fontsize=25)
    #ax.set_xlim([0.0001,0.11])
    ax.legend(loc=3,fontsize=30)
    ax.grid()
    [x.set_linewidth(1.5) for x in ax.spines.values()]
    plt.tight_layout()
    fig.savefig(dname+'/CIFAR10_Auto_Attack_Accuracy_%d.pdf'%norm)
    fig.savefig(dname+'/CIFAR10_Auto_Attack_Accuracy_%d.png'%norm)
def adversarial_example_CNN_CIFAR100():
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    norm = 2
    plot_adversarial_training_accuracy_CIFAR100(ax,norm)
    plot_adversarial_accuracy_CIFAR100(ax,norm)
    if norm==100:
        ax.set_xlabel(r'$l_\infty$ Perturbation $\epsilon$',fontsize=30)
    else:
        ax.set_xlabel(r'$l_2$ Perturbation $\epsilon$',fontsize=30)
    ax.set_ylabel('Test Accuracy',fontsize=30)
    #ax.set_xlim([0.0007,0.047])
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    #ax.set_ylim([0.05,0.71])
    #ax.set_title('Max Evals = %d'%(3*max_evals),fontsize=25)
    #ax.set_xlim([0.0001,0.11])
    ax.legend(loc=3,fontsize=30)
    ax.grid()
    [x.set_linewidth(1.5) for x in ax.spines.values()]
    #plot_adversarial_training_accuracy_PlainCNN(ax)
    #plot_adversarial_training_accuracy_PlainCNN_MT(ax)
    plt.tight_layout()
    fig.savefig(dname+'/Adversarial_Accuracy_Norm_%d_CNN_CIFAR100.pdf'%norm)
#characteristic_time()
#adversarial_timesteps()
#adversarial_example_EP()
#adversarial_example_CNN()
#adversarial_example_CNN_Adv()
#adversarial_example_combined()
#adversarial_example_EPOptimized()
#adversarial_example_Inf_Norm()
#adversarial_confidence()
#adversarial_HSJ()
adversarial_example_CNN_CIFAR100()
CIFAR10_AA()