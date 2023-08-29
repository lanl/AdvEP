import numpy as np
import matplotlib.pyplot as plt
import os
import torch
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
def adv_img_plot():
    eps = np.array([0,0.5,1.0,2.0,3.0,7.0])
    test_image = torch.load(dname+'/Test_Image.pt').numpy()
    test_image = test_image.transpose(1,2,0)
    file_loc = ['/results/EP/mse/2023-05-22/12-28-59_gpu0/','/results/EP/mse/2023-05-22/12-31-48_gpu0/','/results/BPTT/mse/2023-05-22/13-52-18_gpu0/','/results/EP/cel/2023-05-23/16-52-21_gpu0/','/results/BPTT/cel/2023-05-30/15-31-49_gpu0/']
    labels = ['EPRandomMSE','EPSymmMSE','BPTTMSE','EPSymmCE','BPTTCE']
    fig,ax = plt.subplots(len(file_loc),len(eps),figsize=(24,24))
    labeldict = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    for epsi in range(len(eps)):
        for file_iter in range(len(file_loc)):
            axs = ax[file_iter][epsi]
            if(epsi==0):
                diff = (test_image)
                diff -= diff.min()
                diff = diff/(diff.max()-diff.min())
                
                axs.imshow(diff)
                axs.set_ylabel(labels[file_iter],fontsize = 35)
            else:
                filename = file_loc[file_iter]
                try:
                    image = torch.load(dname+filename+'/Adv_Image_Eps_%d.pt'%(epsi-1))
                    image = image.numpy().squeeze(0)
                except FileNotFoundError:
                    continue
                #neurons = torch.load(dname+filename+'/Adv_Label_Eps_%d.pt'%epsi)
                #neurons = neurons.cpu().numpy().squeeze(0)
                axslabel = np.loadtxt(dname+filename+'/Label_Eps_%d.txt'%(epsi-1),dtype=int)
                
                image = image.transpose(1,2,0)
                diff = (image)
                diff -= diff.min()
                diff = diff/(diff.max()-diff.min())
                
                axs.imshow(diff)
                axs.set_xticks([])
                axs.set_yticks([])
                axs.set_title('Pred: %s'%labeldict[axslabel],fontsize=35,c='r',weight='bold')
           
                
            if(file_iter==0):
                axs.annotate(r'$\epsilon=%.1f$'%eps[epsi],
                xy=(0.5, 1.1),
                xytext=(0, 5),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",fontsize = 35,weight='bold'
            )

    plt.tight_layout()
    plt.savefig(dname+'/Adv_Image_Grid.png')
adv_img_plot()