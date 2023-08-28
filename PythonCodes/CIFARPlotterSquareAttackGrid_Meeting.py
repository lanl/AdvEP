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
    eps = np.logspace(np.log10(0.05),np.log10(3),10)
    eps = np.insert(eps,0,1)
    eps = np.sort(eps)
    eps = eps[:-2]
    eps = np.insert(eps,-1,3)
    eps = np.sort(eps)
    img_idx = int(sys.argv[1])
    
    label_cnn_template = 'Labels_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'
    img_cnn_template = 'Adversarial_Images_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'
    label_ep_template = 'White_Last_Layer_Labels_Eps_%d_Attack_T_%d_Predict_T_%d.npy'
    img_ep_template = 'White_Last_Layer_Adversarial_Images_Eps_%d_Attack_T_%d_Predict_T_%d.npy'
    image_hsj_template = 'Adversarial_HSJ_Norm_%d_Images_Train_Eps_%d_Max_Evals_%d_Test_Eps_%d_Seed_%d.npy'
    label_hsj_template = 'Labels_HSJ_Norm_%d_Train_Eps_%d_Max_Evals_%d_Test_Eps_%d_Seed_%d.npy'
    label_cnnrelu_template = 'Labels_Norm_%d_Model_%s_Test_Eps_%d_Seed_%d.npy'
    img_cnnrelu_template = 'Adversarial_Norm_%d_Images_Model_%s_Test_Eps_%d_Seed_%d.npy'
    label_vit_template = 'Norm_%d_Labels_Eps_%d.npy'
    img_vit_template = 'Norm_%d_Adversarial_Images_Eps_%d.npy'
    model_str = 'relu'
    max_evals=1000
    for label_iter in [0,1,2,3]:
        if(label_iter<1): 
            for eps_iter in range(len(eps[5:])):       
                axs = ax[label_iter][eps_iter]
                for file_iter in range(len(file_loc[label_iter])):
                    control =0
                    filename = file_loc[label_iter][file_iter]+img_ep_template%(int(eps[eps_iter+5]*1000),10,250)
                    label_filename = file_loc[label_iter][file_iter]+label_ep_template%(int(eps[eps_iter+5]*1000),10,250)
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
        elif label_iter == 4:
            train_eps = [0.05,0.5,0.75,1]
            max_evals = [1000,20000]
            eval_iter = 0
            test_eps = eps[5:]
            for train_iter in range(len(train_eps)):
                for eps_iter in range(len(test_eps)):       
                    axs = ax[train_iter+1][eps_iter]
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
                        diff = image
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
                        #axs.set_xlabel('Queries: %d'%max_evals[eval_iter],fontsize=35)
                        axs.set_title('Pred: %s'%labeldict[axslabel],fontsize=30,c='r',weight='bold')
                        if(eps_iter==0):
                            axs.set_ylabel(r'CNN$(\epsilon=%.2f)$'%train_eps[train_iter],fontsize = 20)        
                    
    #fig.suptitle(r'$\epsilon =%.2f$'%eps,fontsize=35) 
    print(axslabel)
    plt.tight_layout()
    plt.savefig(dname+'/PGD_Attack_Invert_%d.pdf'%(invert))
adv_img_plot(int(sys.argv[2]))


