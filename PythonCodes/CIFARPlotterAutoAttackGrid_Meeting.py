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
def test_image_plot():

    image = np.load(dname+'/Natural_Images/Images.npy')
    diff = image[int(sys.argv[1])].transpose(1,2,0)
    diff -= diff.min()
    diff = diff/(diff.max()-diff.min())
    
    print(diff.shape)
    plt.imshow(diff)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(dname+'/Test_Image.png')
    return diff
def find_files(file_loc_list):
    param_list = []
    for i in range(len(file_loc_list)):
        for filename in glob.glob(dname+file_loc_list[i]+'Adversarial_Accuracy_A*.txt'):
            fname = filename.replace(dname+file_loc_list[i],'')
            param_list.append(re.findall(r'\d+',fname))
    param_list = np.asarray(param_list,dtype=int)
    param_list = np.unique(param_list,axis=0)
    return param_list
def adv_img_plot(invert=0):
    if(invert==1):
        reference_image = test_image_plot()
    file_loc = [['/results/EP/cel/2023-06-14/17-04-29_gpu0/'],['/ViT-CIFAR/'],
                ['/results/BaselineCNN/cel/Jupyter/MTModel/'],['/results/BaselineCNN/cel/Jupyter/MTModel/'],['/results/BaselineCNN/cel/Jupyter/MTModel/']]
    labels = ['EP CNN','ViT','CNN','CNN','HSJ']
    fig,ax = plt.subplots(4,5,figsize=(18,12))
    labeldict = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    test_eps = np.logspace(np.log10(0.0001),np.log10(0.1),10)  
    #test_eps = np.insert(test_eps,0,1)
    test_eps = np.insert(test_eps,0,8/255)
    eps = np.sort(test_eps)[:-1]
    print(len(eps))
    img_idx = int(sys.argv[1])
    
    label_cnn_template = 'AA_Labels_Norm_100_Square_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'
    img_cnn_template = 'AA_Adversarial_Norm_100_Square_Images_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'
    label_ep_template = 'AA_Norm_100_Square_Labels_Eps_%d_Max_Evals_%d.npy'
    img_ep_template = 'AA_Norm_100_Square_Adversarial_Images_Eps_%d_Max_Evals_%d.npy'
    label_cnnrelu_template = 'AA_Labels_Norm_100_Square_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'
    img_cnnrelu_template = 'AA_Adversarial_Norm_100_Square_Images_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'
    img_vit_template = 'AA_Norm_100_Square_Adversarial_Images_Eps_%d_Patch_4_Augment_1_Seed_%d.npy'
    label_vit_template = 'AA_Norm_100_Square_Labels_Eps_%d_Patch_4_Augment_1_Seed_%d.npy'
    model_str = 'relu'
    max_evals=1000
    for label_iter in [0,1,2,3]:
        if(label_iter<1): 
            for eps_iter in range(len(eps[5:])):       
                axs = ax[label_iter][eps_iter]
                for file_iter in range(len(file_loc[label_iter])):
                    control =0
                    filename = file_loc[label_iter][file_iter]+img_ep_template%(int(eps[eps_iter+5]*1000),1000*1000)
                    label_filename = file_loc[label_iter][file_iter]+label_ep_template%(int(eps[eps_iter+5]*1000),1000*1000)
                    try:
                        image = np.load(dname+filename)
                        control = 1
                    except FileNotFoundError:
                        print(dname+filename)
                        continue

                    axslabel = np.load(dname+label_filename)
                    image = image[img_idx]
                    image = image.transpose(1,2,0)
                    axslabel = axslabel[img_idx]
                    diff = (image)
                    diff -= diff.min()
                    diff = diff/(diff.max()-diff.min())
                    if(invert==1):
                            diff = np.abs(diff-reference_image)
                            diff -= diff.min()
                            diff = diff/(diff.max()-diff.min())
                    else:
                        diff = diff
                    axs.imshow(diff)
                    axs.set_xticks([])
                    axs.set_yticks([])
                    
                    axs.set_title('Pred: %s'%labeldict[axslabel],fontsize=35,c='r',weight='bold')
                    if(label_iter==0):
                        axs.annotate(r'$\epsilon=%.2f$'%eps[5:][eps_iter],
                        xy=(0.5, 1.2),
                        xytext=(0, 5),
                        xycoords="axes fraction",
                        textcoords="offset points",
                        ha="center",
                        va="baseline",fontsize = 40,weight='bold'
                    )
                if(eps_iter==0):
                    axs.set_ylabel(labels[label_iter],fontsize = 35)
        elif label_iter==1:
            for eps_iter in range(len(eps[5:])):       
                axs = ax[1][eps_iter]
                for file_iter in range(len(file_loc[label_iter])):
                    control =0
                    filename = file_loc[label_iter][file_iter]+img_vit_template%(int(eps[5:][eps_iter]*1000),11)
                    print(filename)                                                               
                                                                                    
                                                
                    label_filename = file_loc[label_iter][file_iter]+label_vit_template%(int(eps[5:][eps_iter]*1000),11)
                    try:
                        image = np.load(dname+filename)
                        control = 1
                    except FileNotFoundError:
                        continue

                    axslabel = np.load(dname+label_filename)
                    image = image[img_idx]
                    print(image.shape)
                    image = image.transpose(1,2,0)
                    axslabel = axslabel[img_idx]
                    diff = (image)
                    diff -= diff.min()
                    diff = diff/(diff.max()-diff.min())
                    if(invert==1):
                            diff = np.abs(diff-reference_image)
                            diff -= diff.min()
                            diff = diff/(diff.max()-diff.min())
                    else:
                        diff = diff
                    axs.imshow(diff)
                    axs.set_xticks([])
                    axs.set_yticks([])
                    
                    axs.set_title('Pred: %s'%labeldict[axslabel],fontsize=35,c='r',weight='bold')
                    
                    if(eps_iter==0):
                        axs.set_ylabel(labels[label_iter],fontsize = 35)
        elif label_iter==2:
            for eps_iter in range(len(eps[5:])):       
                axs = ax[label_iter][eps_iter]
                for file_iter in range(len(file_loc[label_iter])):
                    control =0
                    filename = file_loc[label_iter][file_iter]+img_cnnrelu_template%(0,int(eps[5:][eps_iter]*1000),
                                                                                    11)
                                                
                    label_filename = file_loc[label_iter][file_iter]+label_cnnrelu_template%(0,int(eps[5:][eps_iter]*1000),
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
                    diff = (image)
                    diff -= diff.min()
                    diff = diff/(diff.max()-diff.min())
                    if(invert==1):
                            diff = np.abs(diff-reference_image)
                            diff -= diff.min()
                            diff = diff/(diff.max()-diff.min())
                    else:
                        diff = diff
                    axs.imshow(diff)
                    axs.set_xticks([])
                    axs.set_yticks([])
                    
                    axs.set_title('Pred: %s'%labeldict[axslabel],fontsize=35,c='r',weight='bold')
                    
                    if(eps_iter==0):
                        axs.set_ylabel(labels[label_iter],fontsize = 35)      
        elif label_iter==3:
            train_eps = [0,0,1]
            
            for train_iter in range(2,len(train_eps)):
                for eps_iter in range(len(eps[5:])):       
                    axs = ax[label_iter][eps_iter]
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
                        diff = (image)
                        diff -= diff.min()
                        diff = diff/(diff.max()-diff.min())
                        if(invert==1):
                            diff = np.abs(diff-reference_image)
                            diff -= diff.min()
                            diff = diff/(diff.max()-diff.min())
                        else:
                            diff = diff
                        axs.imshow(diff)
                        axs.set_xticks([])
                        axs.set_yticks([])
                        
                        axs.set_title('Pred: %s'%labeldict[axslabel],fontsize=35,c='r',weight='bold')
                        
                        if(eps_iter==0):
                            axs.set_ylabel('%s $(\epsilon=%d)$'%(labels[label_iter],train_eps[train_iter]),fontsize = 35)
 
       
                    
    #fig.suptitle(r'$\epsilon =%.2f$'%eps,fontsize=35) 
    print(axslabel)
    plt.tight_layout()
    plt.savefig(dname+'/AA_Square_Attack_Invert_%d.pdf'%(invert))
adv_img_plot(int(sys.argv[2]))


