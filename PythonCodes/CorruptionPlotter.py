import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cmx
import matplotlib.colors as mcol
import sys
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.weight": "bold"
})

dname = os.getcwd()


def plot_corruption_accuracy(ax,label,data_iter,colno,rowno):
    file_loc = [['/results/EP/cel/2023-06-14/17-04-29_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-02_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-08_gpu0/',
                '/results/EP/cel/2023-06-14/17-06-24_gpu0/',
                '/results/EP/cel/2023-06-15/00-48-27_gpu0/']]
    labels = ['EP CNN']
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
            if(data_iter==colno-1):
                ax.errorbar(np.arange(1,6),adversarial_accuracy_mean[data_iter], yerr = adversarial_accuracy_std[data_iter],label=labels[label_iter],lw=4,c=colors[label_iter])
            else:
                ax.errorbar(np.arange(1,6),adversarial_accuracy_mean[data_iter], yerr = adversarial_accuracy_std[data_iter],label=labels[label_iter],lw=4,c=colors[label_iter])
    if(len(adversarial_accuracies)!=0):
        #ax.set_xlabel(r'Perturbation $\epsilon$',fontsize=25)
        #ax.yaxis.tick_right()
        #ax.set_ylabel('Test Accuracy',fontsize=25)
        #ax.set_xlim([1,100])
        #ax.set_xscale('log')
        ax.xaxis.set_tick_params(labelsize=25)
        ax.yaxis.set_tick_params(labelsize=25)
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
def plot_corruption_accuracy_MT(ax,label,data_iter,colno,rowno):
    train_eps = [0.5,0.75,1]
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
            if(data_iter==colno-1):
                ax.errorbar(np.arange(1,6),adversarial_accuracy_mean[data_iter], yerr = adversarial_accuracy_std[data_iter],lw=4,c=scalarMap.to_rgba(train_eps[eps_iter]),label=r'CNN $(\epsilon=%.2f)$'%train_eps[eps_iter])
            else:
                ax.errorbar(np.arange(1,6),adversarial_accuracy_mean[data_iter], yerr = adversarial_accuracy_std[data_iter],lw=4,c=scalarMap.to_rgba(train_eps[eps_iter]),label=r'CNN $(\epsilon=%.2f)$'%train_eps[eps_iter])
    ax.set_title(setBold(label),fontsize=30)
    #ax.set_ylim([0.00,1])
    if(data_iter%colno==0):
        ax.set_ylabel('Accuracy',fontsize=35)
    if(data_iter//colno==rowno-1):
        ax.set_xlabel('Severity',fontsize=35)
    ax.grid()
    ax.tick_params(axis='both', which='major', labelsize=25)

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
def plot_corrupt_image(ax,label,data_iter,data):
    image = data#.transpose(1,2,0)
    diff = (image)
    diff -= diff.min()
    diff = diff/(diff.max()-diff.min())
    
    ax.imshow(diff)
    ax.set_title(setBold(label),fontsize=30)
    
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.legend(loc=3,fontsize=15)
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
        plot_corruption_accuracy(ax[xpos,ypos],label,i,6,3)
        plot_corruption_accuracy_MT(ax[xpos,ypos],label,i,6,3)
    
    #fig.subplots_adjust(right=0.3, wspace=0.33)
    ax[2][3].legend(loc='center', bbox_to_anchor=(0, -2),fancybox=False, shadow=False,fontsize=20,ncol=4)
    plt.tight_layout()
    fig.savefig(dname+'/Corruption_Accuracy.pdf')

def corruptions_icons():
    fig,ax = plt.subplots(3,3,figsize=(16,15))
    labels = ['Gaussian Noise', 'Motion Blur', 'Zoom Blur',
                'Speckle Noise', 'Contrast', 'Defocus Blur','Fog','JPEG Compression',
                'Pixelate']
    print(len(labels),len(ax[0])*len(ax[:,0]))
    for i in range(len(ax[0])*len(ax[:,0])):
        label = labels[i]
        xpos,ypos = i//len(ax[0]),i%len(ax[0])
        plot_corruption_accuracy(ax[xpos,ypos],label,i,3,3)
        plot_corruption_accuracy_MT(ax[xpos,ypos],label,i,3,3)
    
    #fig.subplots_adjust(right=0.3, wspace=0.33)
    ax[2][1].legend(loc='center', bbox_to_anchor=(0.5, -0.4),fancybox=False, shadow=False,fontsize=30,ncol=4)
    plt.tight_layout()
    fig.savefig(dname+'/Corruption_Accuracy_Legend.pdf')
def corruptions_average():
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    
    plot_corruption_accuracy_average_corruption(ax)
    plot_corruption_accuracy_MT_Average_Corruption(ax)
    ax.set_title('Averaged Over Corruptions',fontsize = 25)
    plt.tight_layout()
    fig.savefig(dname+'/Corruption_Accuracy_Average.png')
def corruptions_image(severity,sample):
    fig,ax = plt.subplots(3,6,figsize=(30,15))
    labels = ['Gaussian Noise', 'Shot Noise', 'Motion Blur', 'Zoom Blur',
                'Spatter', 'Brightness' ,'Speckle Noise', 'Gaussian Blur', 'Snow',
                'Contrast', 'Defocus Blur','Elastic Transform','Fog','Glass Blur','Impulse Noise','JPEG Compression',
                'Pixelate','Saturate']
    fname = ['gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur',
                'spatter', 'brightness' ,'speckle_noise', 'gaussian_blur', 'snow',
                'contrast', 'defocus_blur','elastic_transform','fog','glass_blur','impulse_noise','jpeg_compression',
                'pixelate','saturate','frost']
    c_p_dir = '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/vision-greg3/CIFAR-10-C'
    print(len(labels),len(ax[0])*len(ax[:,0]))
    for i in range(len(ax[0])*len(ax[:,0])):
        label = labels[i]
        dataset = np.load(os.path.join(c_p_dir, fname[i] + '.npy'))
        xpos,ypos = i//len(ax[0]),i%len(ax[0])
        plot_corrupt_image(ax[xpos,ypos],label,i,dataset[severity*10000+sample])
    plt.tight_layout()
    fig.savefig(dname+'/Corrupted_Image_Severity_%d_Sample_%d.pdf'%(severity+1,sample))
def corruptions_icons_images(severity,sample):
    fig,ax = plt.subplots(3,3,figsize=(16,15))
    labels = ['Gaussian Noise', 'Motion Blur', 'Zoom Blur',
                'Speckle Noise', 'Contrast', 'Defocus Blur','Fog','JPEG Compression',
                'Pixelate']
    fname = ['gaussian_noise', 'motion_blur', 'zoom_blur',
                'speckle_noise', 'contrast', 'defocus_blur','fog','jpeg_compression',
                'pixelate',]
    c_p_dir = '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/vision-greg3/CIFAR-10-C'
    print(len(labels),len(ax[0])*len(ax[:,0]))
    for i in range(len(ax[0])*len(ax[:,0])):
        label = labels[i]
        dataset = np.load(os.path.join(c_p_dir, fname[i] + '.npy'))
        xpos,ypos = i//len(ax[0]),i%len(ax[0])
        plot_corrupt_image(ax[xpos,ypos],label,i,dataset[severity*10000+sample])
    plt.tight_layout()
    fig.savefig(dname+'/Corrupted_Image_Severity_%d_Sample_%d.pdf'%(severity+1,sample))
#corruptions()
corruptions_icons()
corruptions_icons_images(severity = int(sys.argv[1]),sample=7)
#corruptions_average()
#corruptions_image(severity=int(sys.argv[1]),sample=7)
#adversarial_example()