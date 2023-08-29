import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cmx
import matplotlib.colors as mcol
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.weight": "bold"
})

dname = os.getcwd()
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
    
    file_loc = ['/results/EP/mse/2023-05-22/12-31-48_gpu0/',
                '/results/EP/mse/2023-05-22/12-28-59_gpu0/',
                '/results/EP/cel/2023-05-23/16-52-21_gpu0/',
                '/results/EP/cel/2023-05-23/15-40-03_gpu0/',
                '/results/BPTT/mse/2023-05-22/13-52-18_gpu0/',
                '/results/BPTT/cel/2023-05-30/15-31-49_gpu0/',
                '/results/BaselineCNN/cel/Jupyter/','/results/BaselineCNN/cel/Jupyter/']
    labels = ['Alg: EP With Symmetric Gradient, Loss: MSE',
              'Alg: EP With Random Sign, Loss: MSE',
              'Alg: EP With Symmetric Gradient, Loss: CE',
              'Alg: EP With Symmetric Gradient, Loss: CE',
              'Alg: BPTT, Loss: MSE',
              'Alg: BPTT, Loss: CE',
              'Vanilla CNN (HardSigmoid), Loss: CE','Vanilla CNN (ReLU), Loss: CE']
    colors = ['k','k','k','k','k','k','g','c']
    for file_iter in range(2,len(file_loc)):
        if(file_iter>=3 and file_iter<=5):
            continue
        filename = file_loc[file_iter]
        try:
            if(file_iter==7):
                #print(labels[file_iter],dname+filename+'Adversarial_Accuracy_UnNormalized_ReLu.txt')
                adversarial_accuracy = np.loadtxt(dname+filename+'Adversarial_Accuracy_UnNormalized_Relu.txt')
                #print(labels[file_iter],dname+filename+'Adversarial_Accuracy_UnNormalized_ReLu.txt')
            elif(file_iter==6):
                adversarial_accuracy = np.loadtxt(dname+filename+'Adversarial_Accuracy_UnNormalized_HardSigmoid.txt')
            else:
                adversarial_accuracy = np.loadtxt(dname+filename+'Adversarial_Accuracy_Diff_Package.txt')
        except OSError:
            
            continue 
        ax.plot(adversarial_accuracy[:,0],adversarial_accuracy[:,2],'.-',label=labels[file_iter],lw=2,c=colors[file_iter])
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

def plot_adversarial_accuracy_MT(ax):
    
    file_loc = ['/results/EP/mse/2023-05-22/12-31-48_gpu0/',
                '/results/EP/mse/2023-05-22/12-28-59_gpu0/',
                '/results/EP/cel/2023-05-23/16-52-21_gpu0/',
                '/results/EP/cel/2023-05-23/15-40-03_gpu0/',
                '/results/BPTT/mse/2023-05-22/13-52-18_gpu0/',
                '/results/BPTT/cel/2023-05-30/15-31-49_gpu0/',
                '/results/BaselineCNN/cel/Jupyter/MTModel/','/results/BaselineCNN/cel/Jupyter/MTModel/']
    labels = ['Alg: EP With Symmetric Gradient, Loss: MSE',
              'Alg: EP With Random Sign, Loss: MSE',
              'Alg: EP With Symmetric Gradient, Loss: CE',
              'Alg: EP With Symmetric Gradient, Loss: CE',
              'Alg: BPTT, Loss: MSE',
              'Alg: BPTT, Loss: CE',
              'Vanilla CNN (HardSigmoid), Loss: CE','Vanilla CNN (ReLU), Loss: CE']
    colors = ['k','k','k','k','k','k','g','c']
    for file_iter in range(2,len(file_loc)):
        if(file_iter>=3 and file_iter<=5):
            continue
        filename = file_loc[file_iter]
        try:
            if(file_iter==7):
                #print(labels[file_iter],dname+filename+'Adversarial_Accuracy_UnNormalized_ReLu.txt')
                adversarial_accuracy = np.loadtxt(dname+filename+'Adversarial_Accuracy_relu.txt')
                #print(labels[file_iter],dname+filename+'Adversarial_Accuracy_UnNormalized_ReLu.txt')
            elif(file_iter==6):
                adversarial_accuracy = np.loadtxt(dname+filename+'Adversarial_Accuracy_hardsigmoid.txt')
            else:
                adversarial_accuracy = np.loadtxt(dname+filename+'Adversarial_Accuracy_Diff_Package.txt')
        except OSError:
            
            continue 
        ax.plot(adversarial_accuracy[:,0],adversarial_accuracy[:,2],'.-',label=labels[file_iter],lw=2,c=colors[file_iter])
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

def plot_adversarial_training_accuracy_PlainCNN(ax):
    train_eps = [0.05,0.1,0.25,0.5,0.75,1]
    #train_eps = [0.5,1]
    cm = plt.get_cmap('jet')
    cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    cNorm = mcol.Normalize(vmin=min(train_eps), vmax=max(train_eps))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    for eps_iter in range(1,len(train_eps),2):
        adversarial_training = np.loadtxt(dname+'/results/BaselineCNN/cel/Jupyter/Adversarial_Accuracy_UnNormalized_Eps_%d.txt'%(int(train_eps[eps_iter]*1000)))
        ax.plot(adversarial_training[:,0],adversarial_training[:,2],'.-',label=r'VCNN-$\epsilon=%.2f$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
    ax.legend(loc=3,fontsize=15)
def plot_adversarial_training_accuracy_PlainCNN_MT(ax):
    train_eps = [0.05,0.5,0.75,1]
    cm = plt.get_cmap('jet')
    cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    cNorm = mcol.Normalize(vmin=min(train_eps), vmax=max(train_eps))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    for eps_iter in range(len(train_eps)):
        adversarial_training = np.loadtxt(dname+'/results/BaselineCNN/cel/Jupyter/MTModel/Adversarial_Accuracy_Eps_%d.txt'%(int(train_eps[eps_iter]*1000)))
        #ax.plot(adversarial_training[:,0],adversarial_training[:,2],'.-',label=r'VCNN-$\epsilon=%.2f$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
        ax.plot(adversarial_training[:,0],adversarial_training[:,2],'.-',label=r'VCNN ($\epsilon=%.2f)$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
    ax.legend(loc=3,fontsize=15)
def plot_corruption_accuracy(ax,label,data_iter):
    file_loc = [['/results/EP/cel/2023-06-14/17-04-29_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-02_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-08_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-24_gpu0/',
                '/results/EP/cel/2023-06-15/00-48-27_gpu0/']]
    labels = ['EP With Symmetric Gradient, Loss: CE']
    colors = ['k','m','g','c']
    for label_iter in range(len(labels)):
        adversarial_accuracies = []
        for filename in file_loc[label_iter]:
            try:
                adversarial_accuracy = np.loadtxt(dname+filename+'Corruption_Accuracy.txt')
                adversarial_accuracies.append(adversarial_accuracy)
            except OSError:
                continue 
        if(len(adversarial_accuracies)!=0):
            adversarial_accuracy_mean = np.mean(np.asarray(adversarial_accuracies),axis=0)
            adversarial_accuracy_std = np.std(np.asarray(adversarial_accuracies),axis=0)
            ax.errorbar(np.arange(1,6),adversarial_accuracy_mean[data_iter], yerr = adversarial_accuracy_std[data_iter],label=labels[label_iter],lw=2,c=colors[label_iter])
    if(len(adversarial_accuracies)!=0):
        #ax.set_xlabel(r'Perturbation $\epsilon$',fontsize=25)
        #ax.yaxis.tick_right()
        #ax.set_ylabel('Test Accuracy',fontsize=25)
        #ax.set_xlim([1,100])
        #ax.set_xscale('log')
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        #ax.set_ylim([0.05,0.95])
        #ax.set_xlim([0.05,3])
        #ax.legend(loc=3,fontsize=15)
        
        [x.set_linewidth(1.5) for x in ax.spines.values()]
def plot_corruption_accuracy_average_corruption(ax):
    file_loc = [['/results/EP/cel/2023-06-14/17-04-29_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-02_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-08_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-24_gpu0/',
                '/results/EP/cel/2023-06-15/00-48-27_gpu0/']]
    labels = ['EP With Symmetric Gradient, Loss: CE']
    colors = ['k','m','g','c']
    for label_iter in range(len(labels)):
        adversarial_accuracies = []
        for filename in file_loc[label_iter]:
            try:
                adversarial_accuracy = np.loadtxt(dname+filename+'Corruption_Accuracy.txt')
                adversarial_accuracies.append(adversarial_accuracy)
            except OSError:
                continue 
        if(len(adversarial_accuracies)!=0):
            adversarial_accuracy_mean = np.mean(np.asarray(adversarial_accuracies),axis=0)
            adversarial_accuracy_std = np.std(np.asarray(adversarial_accuracies),axis=0)
            ax.errorbar(np.arange(1,6),np.mean(adversarial_accuracy_mean,axis=0), yerr = np.std(adversarial_accuracy_mean,axis=0),label=labels[label_iter],lw=2,c=colors[label_iter])
    if(len(adversarial_accuracies)!=0):
        #ax.set_xlabel(r'Perturbation $\epsilon$',fontsize=25)
        #ax.yaxis.tick_right()
        #ax.set_ylabel('Test Accuracy',fontsize=25)
        #ax.set_xlim([1,100])
        #ax.set_xscale('log')
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        #ax.set_ylim([0.05,0.95])
        #ax.set_xlim([0.05,3])
        #ax.legend(loc=3,fontsize=15)
        
        [x.set_linewidth(1.5) for x in ax.spines.values()]
def plot_corruption_accuracy_MT(ax,label,data_iter):
    train_eps = [0.05,0.5,0.75,1]
    cm = plt.get_cmap('jet')
    cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    cNorm = mcol.Normalize(vmin=min(train_eps), vmax=max(train_eps))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    for eps_iter in range(len(train_eps)):
    
        adversarial_accuracies = []
        for seed in [11,13,17,19,31]:
            adversarial_accuracy = np.loadtxt(dname+'/results/BaselineCNN/cel/Jupyter/MTModel/Corruption_Accuracy_Eps_%d_Seed_%d.txt'%(int(train_eps[eps_iter]*1000),seed))
            adversarial_accuracies.append(adversarial_accuracy)
            #ax.plot(adversarial_training[:,0],adversarial_training[:,2],'.-',label=r'VCNN-$\epsilon=%.2f$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
            #ax.plot(np.arange(1,6),adversarial_training[2],'.-',label=r'VCNN ($\epsilon=%.2f)$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
        if(len(adversarial_accuracies)!=0):
            adversarial_accuracy_mean = np.mean(np.asarray(adversarial_accuracies),axis=0)
            adversarial_accuracy_std = np.std(np.asarray(adversarial_accuracies),axis=0)
            ax.errorbar(np.arange(1,6),adversarial_accuracy_mean[data_iter], yerr = adversarial_accuracy_std[data_iter],lw=2,c=scalarMap.to_rgba(train_eps[eps_iter]))
    ax.set_title(setBold(label),fontsize=30)
    #ax.set_ylim([0.00,1])
    if(data_iter%6==0):
        ax.set_ylabel('Accuracy',fontsize=25)
    if(data_iter//6==2):
        ax.set_xlabel('Severity',fontsize=25)
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=18)

    #ax.legend(loc=3,fontsize=15)
def plot_corruption_accuracy_MT_Average_Corruption(ax):
    train_eps = [0.05,0.5,0.75,1]
    cm = plt.get_cmap('jet')
    cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    cNorm = mcol.Normalize(vmin=min(train_eps), vmax=max(train_eps))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    for eps_iter in range(len(train_eps)):
    
        adversarial_accuracies = []
        for seed in [11,13,17,19,31]:
            adversarial_accuracy = np.loadtxt(dname+'/results/BaselineCNN/cel/Jupyter/MTModel/Corruption_Accuracy_Eps_%d_Seed_%d.txt'%(int(train_eps[eps_iter]*1000),seed))
            adversarial_accuracies.append(adversarial_accuracy)
            #ax.plot(adversarial_training[:,0],adversarial_training[:,2],'.-',label=r'VCNN-$\epsilon=%.2f$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
            #ax.plot(np.arange(1,6),adversarial_training[2],'.-',label=r'VCNN ($\epsilon=%.2f)$'%train_eps[eps_iter],c=scalarMap.to_rgba(train_eps[eps_iter]),lw=2)
        if(len(adversarial_accuracies)!=0):
            adversarial_accuracy_mean = np.mean(np.asarray(adversarial_accuracies),axis=0)
            adversarial_accuracy_std = np.std(np.asarray(adversarial_accuracies),axis=0)
            ax.errorbar(np.arange(1,6),np.mean(adversarial_accuracy_mean,axis=0), yerr = np.std(adversarial_accuracy_mean,axis=0),label=r'VCNN ($\epsilon=%.2f)$'%train_eps[eps_iter],
                        lw=2,c=scalarMap.to_rgba(train_eps[eps_iter]))
    ax.set_ylabel('Accuracy',fontsize=25)
    ax.set_xlabel('Severity',fontsize=25)
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=18)

    ax.legend(loc=3,fontsize=15)
def setBold(txt): return r"$\bf{" + str(txt) + "}$"
def prelim_figure():

    fig,ax = plt.subplots(1,2,figsize=(16,8))
    plot_characteristic_time(ax[0])
    plot_adversarial_accuracy(ax[1])
    plot_adversarial_training_accuracy_PlainCNN(ax[1])
    plt.tight_layout()
    fig.savefig(dname+'/Characteristic_Time3.png')
def adversarial_example():
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    #plot_characteristic_time(ax[0])
    plot_adversarial_accuracy_MT(ax)
    #plot_adversarial_training_accuracy_PlainCNN(ax)
    plot_adversarial_training_accuracy_PlainCNN_MT(ax)
    plt.tight_layout()
    fig.savefig(dname+'/Adversarial_Accuracy.png')
def corruptions():
    fig,ax = plt.subplots(3,6,figsize=(30,15))
    labels = ['Gaussian Noise', 'Shot Noise', 'Motion Blur', 'Zoom Blur',
                'Spatter', 'Brightness' ,'Speckle Noise', 'Gaussian Blur', 'Snow',
                'Contrast', 'Defocus Blur','Elastic Transform','Fog','Glass Blur','Impulse Noise','JPEG Compression',
                'Pixelate','Saturate']
    print(len(labels),len(ax[0])*len(ax[:,0]))
    for i in range(len(ax[0])*len(ax[:,0])):
        label = labels[i]
        xpos,ypos = i//len(ax[0]),i%len(ax[0])
        plot_corruption_accuracy(ax[xpos,ypos],label,i)
        plot_corruption_accuracy_MT(ax[xpos,ypos],label,i)
    plt.tight_layout()
    fig.savefig(dname+'/Corruption_Accuracy.png')
def corruptions_average():
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    
    plot_corruption_accuracy_average_corruption(ax)
    plot_corruption_accuracy_MT_Average_Corruption(ax)
    ax.set_title('Averaged Over Corruptions',fontsize = 25)
    plt.tight_layout()
    fig.savefig(dname+'/Corruption_Accuracy_Average.png')
corruptions()
corruptions_average()
#adversarial_example()