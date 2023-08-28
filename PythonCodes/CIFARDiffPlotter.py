import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import glob
import re
import sys
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.weight": "bold"
})

dname = os.getcwd()
def test_image_recover():
    eps = np.logspace(np.log10(0.0001),np.log10(0.1),10)
    img_idx = int(sys.argv[1])
    img_cnn_template = 'Adversarial_Norm_100_Images_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'
    filename = dname+'/results/BaselineCNN/cel/Jupyter/MTModel/'+img_cnn_template%(int(0.05*1000),int(eps[0]*1000),11)
    image = np.load(filename)
    image = image[img_idx]
    image = image.transpose(1,2,0)
    return image
def adv_img_limits():
    file_loc = [['/results/EP/cel/2023-06-14/17-04-29_gpu0/'],
                ['/results/BaselineCNN/cel/Jupyter/MTModel/']]
    labels = ['EP','CNN']
    
    labeldict = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    eps = np.logspace(np.log10(0.05),np.log10(3),10)
    eps = np.insert(eps,0,1)
    eps = np.sort(eps)
    img_idx = int(sys.argv[1])
    limits = []
    test_image = test_image_recover()
    label_cnn_template = 'Labels_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'
    img_cnn_template = 'Adversarial_Images_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'
    for label_iter in range(2):
        if(label_iter==0):
            img_template = 'White_Last_Layer_Adversarial_Images_Eps_%d_Attack_T_%d_Predict_T_%d.npy'
            label_template = 'White_Last_Layer_Labels_Eps_%d_Attack_T_%d_Predict_T_%d.npy'
            attack_timesteps = [10,29,70]
            predict_timesteps = [250]
            for attack_iter in range(len(attack_timesteps)):
                for eps_iter in range(len(eps[5:])):       
                    for file_iter in range(len(file_loc[label_iter])):
                        control =0
                        filename = file_loc[label_iter][file_iter]+img_template%(int(eps[5:][eps_iter]*1000),
                                                    attack_timesteps[attack_iter],predict_timesteps[0])
                        label_filename = file_loc[label_iter][file_iter]+label_template%(int(eps[5:][eps_iter]*1000),
                                                    attack_timesteps[attack_iter],predict_timesteps[0])
                        try:
                            image = np.load(dname+filename)
                            control = 1
                        except FileNotFoundError:
                            continue

                        axslabel = np.load(dname+label_filename)
                        image = image[img_idx]
                        image = image.transpose(1,2,0)
                        diff = np.abs(image-test_image)
                        diff -= diff.min()
                        image = diff/(diff.max()-diff.min())
                        img_max = image.max()
                        img_min = image.min()
                        limits.append([img_min,img_max])
                        
                    
        else:
            train_eps = [0,0,0,0.05,0.5,1]
            
            for train_iter in range(3,len(train_eps)):
                for eps_iter in range(len(eps[5:])):       
                    for file_iter in range(len(file_loc[label_iter])):
                        control =0
                        filename = file_loc[label_iter][file_iter]+img_cnn_template%(int(train_eps[train_iter]*1000),
                                                                                     int(eps[5:][eps_iter]*1000),
                                                                                     11)
                                                    
                        label_filename = file_loc[label_iter][file_iter]+label_cnn_template%(int(train_eps[train_iter]*1000),
                                                                                     int(eps[5:][eps_iter]*1000),
                                                                                     11)
                        try:
                            image = np.load(dname+filename)
                            control = 1
                        except FileNotFoundError:
                            continue

                        
                        image = image[img_idx]
                        image = image.transpose(1,2,0)
                        diff = np.abs(image-test_image)
                        diff -= diff.min()
                        image = diff/(diff.max()-diff.min())
                        img_max = image.max()
                        img_min = image.min()
                        limits.append([img_min,img_max])
    limits = np.asarray(limits)
    return np.min(limits[:,0]), np.max(limits[:,1])
def find_files(file_loc_list):
    param_list = []
    for i in range(len(file_loc_list)):
        for filename in glob.glob(dname+file_loc_list[i]+'Adversarial_Accuracy_A*.txt'):
            fname = filename.replace(dname+file_loc_list[i],'')
            param_list.append(re.findall(r'\d+',fname))
    param_list = np.asarray(param_list,dtype=int)
    param_list = np.unique(param_list,axis=0)
    return param_list
def adv_img_plot():
    file_loc = [['/results/EP/cel/2023-06-14/17-04-29_gpu0/'],
                ['/results/BaselineCNN/cel/Jupyter/MTModel/']]
    labels = ['EP','CNN']
    fig,ax = plt.subplots(6,6,figsize=(24,24))
    labeldict = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    eps = np.logspace(np.log10(0.05),np.log10(3),10)
    eps = np.insert(eps,0,1)
    eps = np.sort(eps)
    img_idx = int(sys.argv[1])
    img_min, img_max = adv_img_limits()
    test_image = test_image_recover()
    label_cnn_template = 'Labels_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'
    img_cnn_template = 'Adversarial_Images_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'
    for label_iter in range(2):
        if(label_iter==0):
            img_template = 'White_Last_Layer_Adversarial_Images_Eps_%d_Attack_T_%d_Predict_T_%d.npy'
            label_template = 'White_Last_Layer_Labels_Eps_%d_Attack_T_%d_Predict_T_%d.npy'
            attack_timesteps = [10,29,70]
            predict_timesteps = [250]
            for attack_iter in range(len(attack_timesteps)):
                for eps_iter in range(len(eps[5:])):       
                    axs = ax[eps_iter][attack_iter]
                    for file_iter in range(len(file_loc[label_iter])):
                        control =0
                        filename = file_loc[label_iter][file_iter]+img_template%(int(eps[5:][eps_iter]*1000),
                                                    attack_timesteps[attack_iter],predict_timesteps[0])
                        label_filename = file_loc[label_iter][file_iter]+label_template%(int(eps[5:][eps_iter]*1000),
                                                    attack_timesteps[attack_iter],predict_timesteps[0])
                        try:
                            image = np.load(dname+filename)
                            control = 1
                        except FileNotFoundError:
                            continue

                        axslabel = np.load(dname+label_filename)
                        image = image[img_idx]
                        image = image.transpose(1,2,0)
                        axslabel = axslabel[img_idx]
                        diff = np.abs(image-test_image)
                        diff -= diff.min()
                        diff = diff/(diff.max()-diff.min())
                        
                        axs.imshow(diff,vmin=img_min,vmax=img_max)
                        axs.set_xticks([])
                        axs.set_yticks([])
                        
                        axs.set_title('Pred: %s'%labeldict[axslabel],fontsize=35,c='r',weight='bold')
                        if(attack_iter==0):
                            """ axs.annotate(r'$\epsilon=%.2f$'%eps[5:][eps_iter],
                            xy=(0.5, 1.1),
                            xytext=(0, 5),
                            xycoords="axes fraction",
                            textcoords="offset points",
                            ha="baseline",
                            va="center",fontsize = 40,weight='bold'
                        ) """
                            axs.set_ylabel(r'$\epsilon=%.2f$'%eps[5:][eps_iter],fontsize=35)
                    if(eps_iter==len(eps[5:])-1):
                        axs.set_xlabel(labels[label_iter]+r' Att $T=%d$'%attack_timesteps[attack_iter],fontsize = 35)
                    
        else:
            train_eps = [0,0,0,0.05,0.5,1]
            
            for train_iter in range(3,len(train_eps)):
                for eps_iter in range(len(eps[5:])):       
                    axs = ax[eps_iter][train_iter]
                    for file_iter in range(len(file_loc[label_iter])):
                        control =0
                        filename = file_loc[label_iter][file_iter]+img_cnn_template%(int(train_eps[train_iter]*1000),
                                                                                     int(eps[5:][eps_iter]*1000),
                                                                                     11)
                                                    
                        label_filename = file_loc[label_iter][file_iter]+label_cnn_template%(int(train_eps[train_iter]*1000),
                                                                                     int(eps[5:][eps_iter]*1000),
                                                                                     11)
                        try:
                            image = np.load(dname+filename)
                            control = 1
                        except FileNotFoundError:
                            continue

                        axslabel = np.load(dname+label_filename)
                        image = image[img_idx]
                        image = image.transpose(1,2,0)
                        axslabel = axslabel[img_idx]
                        diff = np.abs(image-test_image)
                        diff -= diff.min()
                        diff = diff/(diff.max()-diff.min())
                        
                        axs.imshow(diff,vmin=img_min,vmax=img_max)
                        axs.set_xticks([])
                        axs.set_yticks([])
                        
                        axs.set_title('Pred: %s'%labeldict[axslabel],fontsize=35,c='r',weight='bold')
                        if(attack_iter==0 and label_iter==0):
                            axs.annotate(r'$\epsilon=%.2f$'%eps[5:][eps_iter],
                            xy=(0.5, 1.1),
                            xytext=(0, 5),
                            xycoords="axes fraction",
                            textcoords="offset points",
                            ha="center",
                            va="baseline",fontsize = 40,weight='bold'
                        )
                        if(eps_iter==len(eps[5:])-1):
                            axs.set_xlabel(labels[label_iter]+r'Train $\epsilon=%.1f$'%train_eps[train_iter],fontsize = 35)
                        
               
    #fig.suptitle(r'$\epsilon =%.2f$'%eps,fontsize=35) 
    print(axslabel)
    fig.tight_layout()
    plt.savefig(dname+'/Adv_Image_Diff_Abs.png')
def test_image_plot():
    filename = dname+'/Natural_Images/Images.npy'
    img_idx = int(sys.argv[1])
    image = np.load(filename)
    labeldict = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    label = np.load(dname+'/Natural_Images/Labels.npy')
    print(labeldict[label[img_idx]])
    image = image[img_idx]
    image = image.transpose(1,2,0)
    image -= image.min()
    image = image/(image.max()-image.min())
    fig,ax = plt.subplots(1,1,figsize=(4,4))
    plt.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(dname+'/Frog_Image.png')
test_image_plot()
#adv_img_plot()


