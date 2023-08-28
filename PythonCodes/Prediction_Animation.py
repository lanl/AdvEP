import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import glob
import re
from matplotlib.animation import FuncAnimation
import sys
from matplotlib import animation, rc
from IPython.display import HTML, Image
rc('animation', html='html5')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
})

dname = os.getcwd()

fileloc = '/results/EP/cel/2023-06-14/17-04-29_gpu0/'
predictions_filename = fileloc+'Characteristic_Predictions.txt'
logits_filename = fileloc+'Characteristic_Logits.npy'
natural_image_loc = dname+'/Natural_Images/Images.npy'
natural_labels_loc = dname+'/Natural_Images/Labels.npy'
predictions = np.loadtxt(dname+predictions_filename)
logits = np.load(dname+logits_filename)
first_correct = np.argmax(predictions,axis=1)
natural_images = np.load(natural_image_loc)
natural_labels = np.load(natural_labels_loc)
idx = np.argwhere(first_correct>100).flatten()[2:]
print(idx,first_correct[idx])
for img_idx in idx:
    img_logits = logits[img_idx]
    fig, ax = plt.subplots(1,2,figsize=(15,8),tight_layout=True)
    #fig, ax = plt.subplots(1,2,figsize=(15,8))
    diff = natural_images[img_idx].transpose(1,2,0)
    diff -= diff.min()
    diff = diff/(diff.max()-diff.min())
    ax[0].imshow(diff, aspect="auto")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    labeldict = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    rects = ax[1].bar(labeldict, [.1]*10, align='center')
    i=0
    if(i<first_correct[img_idx]):
            ax[1].set_title('t=%d'%i,c='r',fontsize=25)
    else:
        ax[1].set_title('t=%d'%i,c='k',fontsize=25)
    plt.setp(ax[1].get_xticklabels(), rotation=45, horizontalalignment='right',fontsize=25)
    ax[1].set_ylim([0,1.0])
    def init():
        #line.set_data([], [])
        return rects
    def animate(i):

        x = labeldict
        y = img_logits[i]
        for rect, h,xlabel in zip(rects, y, x):
            rect.set_height(h)
        #line.set_data(x, y)
        if(i<first_correct[img_idx]):
            ax[1].set_title('t=%d'%i,c='r',fontsize=25)
        else:
            ax[1].set_title('t=%d'%i,c='k',fontsize=25)
        
        return rects
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=250, interval=20, blit=True)
    anim.save(dname+'/Predictions_Bar_Plot/Image_%d.gif'%img_idx, writer='imagemagick', fps=30,dpi=300)
