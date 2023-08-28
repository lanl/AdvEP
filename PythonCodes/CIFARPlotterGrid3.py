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

    image = torch.load(dname+'/Test_Image.pt').numpy()
    image = image.transpose(1,2,0)
    print(image.shape)
    plt.imshow(image)
    plt.savefig(dname+'/Test_Image.png')
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
    file_loc = [['/results/EP/cel/2023-06-14/17-04-29_gpu0/'],['/ViT-CIFAR/'],
                ['/results/BaselineCNN/cel/Jupyter/MTModel/'],['/results/BaselineCNN/cel/Jupyter/MTModel/'],['/results/BaselineCNN/cel/Jupyter/MTModel/']]
    labels = ['EP','ViT','CNN','AdvCNN','HSJ']
    fig,ax = plt.subplots(5,6,figsize=(24,12))
    labeldict = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    eps = np.logspace(np.log10(0.05),np.log10(3),10)
    eps = np.insert(eps,0,1)
    eps = np.sort(eps)
    img_idx = int(sys.argv[1])
    
    label_cnn_template = 'Labels_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'
    img_cnn_template = 'Adversarial_Images_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'
    image_hsj_template = 'Adversarial_HSJ_Norm_%d_Images_Train_Eps_%d_Max_Evals_%d_Test_Eps_%d_Seed_%d.npy'
    label_hsj_template = 'Labels_HSJ_Norm_%d_Train_Eps_%d_Max_Evals_%d_Test_Eps_%d_Seed_%d.npy'
    label_cnnrelu_template = 'Labels_Norm_%d_Model_%s_Test_Eps_%d_Seed_%d.npy'
    img_cnnrelu_template = 'Adversarial_Norm_%d_Images_Model_%s_Test_Eps_%d_Seed_%d.npy'
    label_vit_template = 'Norm_%d_Labels_Eps_%d.npy'
    img_vit_template = 'Norm_%d_Adversarial_Images_Eps_%d.npy'
    model_str = 'relu'
    for label_iter in range(len(labels)):
        if(label_iter<1): 
            img_template = 'White_Last_Layer_Adversarial_Images_Eps_%d_Attack_T_%d_Predict_T_%d.npy'
            label_template = 'White_Last_Layer_Labels_Eps_%d_Attack_T_%d_Predict_T_%d.npy'
            attack_timesteps = [10]
            predict_timesteps = [250]
            for attack_iter in range(len(attack_timesteps)):
                for eps_iter in range(len(eps[5:])):       
                    axs = ax[label_iter][eps_iter]
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
                            print(dname+filename)
                            continue

                        axslabel = np.load(dname+label_filename)
                        image = image[img_idx]
                        image = image.transpose(1,2,0)
                        axslabel = axslabel[img_idx]
                        diff = (image)
                        diff -= diff.min()
                        diff = diff/(diff.max()-diff.min())
                        
                        axs.imshow(diff)
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
                    if(eps_iter==0):
                        axs.set_ylabel(labels[label_iter],fontsize = 35)
        elif label_iter==1:
            for eps_iter in range(len(eps[5:])):       
                axs = ax[1][eps_iter]
                for file_iter in range(len(file_loc[label_iter])):
                    control =0
                    filename = file_loc[label_iter][file_iter]+img_vit_template%(2,int(eps[5:][eps_iter]*1000))
                    print(filename)                                                               
                                                                                    
                                                
                    label_filename = file_loc[label_iter][file_iter]+label_vit_template%(2,int(eps[5:][eps_iter]*1000))
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
                    
                    axs.imshow(diff)
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
                    if(eps_iter==0):
                        axs.set_ylabel(labels[label_iter],fontsize = 35)
        elif label_iter==2:
            for eps_iter in range(len(eps[5:])):       
                axs = ax[label_iter][eps_iter]
                for file_iter in range(len(file_loc[label_iter])):
                    control =0
                    filename = file_loc[label_iter][file_iter]+img_cnnrelu_template%(2,model_str,
                                                                                    int(eps[5:][eps_iter]*1000),
                                                                                    11)
                                                
                    label_filename = file_loc[label_iter][file_iter]+label_cnnrelu_template%(2,model_str,
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
                    
                    axs.imshow(diff)
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
                    if(eps_iter==0):
                        axs.set_ylabel(labels[label_iter],fontsize = 35)      
        elif label_iter==3:
            train_eps = [0,0,0.75]
            
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
                        
                        axs.imshow(diff)
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
                        if(eps_iter==0):
                            axs.set_ylabel(labels[label_iter],fontsize = 35)
        elif label_iter == 4:
            train_eps = [0,0,0.05]
            max_evals = [1000,20000]
            eval_iter = 1
            test_eps = eps[5:]
            for train_iter in range(2,len(train_eps)):
                for eps_iter in range(len(test_eps)):       
                    axs = ax[label_iter][eps_iter]
                    for file_iter in range(len(file_loc[label_iter])):
                        control =0
                        filename = file_loc[label_iter][file_iter]+image_hsj_template%(2,int(train_eps[train_iter]*1000),
                                                                                     int(max_evals[eval_iter]*1000),int(test_eps[eps_iter]*1000), 
                                                                                     11)
                        print(filename)
                        label_filename = file_loc[label_iter][file_iter]+label_hsj_template%(2,int(train_eps[train_iter]*1000),
                                                                                     int(max_evals[eval_iter]*1000),int(test_eps[eps_iter]*1000),
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
                        
                        axs.imshow(diff)
                        axs.set_xticks([])
                        axs.set_yticks([])
                        axs.set_xlabel('Queries: %d'%max_evals[eval_iter],fontsize=35)
                        axs.set_title('Pred: %s'%labeldict[axslabel],fontsize=35,c='r',weight='bold')
                        if(eval_iter==0):
                            axs.set_ylabel(labels[label_iter],fontsize = 35)        
                    
    #fig.suptitle(r'$\epsilon =%.2f$'%eps,fontsize=35) 
    print(axslabel)
    fig.tight_layout()
    plt.savefig(dname+'/HSJ_Attack_T_%d_Predict_T_%d.png'%(attack_timesteps[0],predict_timesteps[0]))
adv_img_plot()


