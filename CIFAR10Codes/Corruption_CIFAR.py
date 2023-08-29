import numpy as np
import os
import argparse
import time
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F

BATCH_SIZE = 200
EPOCHS = 45

label_dir = '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/vision-greg3/CIFAR-10-C'
labels = torch.from_numpy(np.load(os.path.join(label_dir, 'labels.npy')))
def isinteger(x):
    return np.equal(np.mod(x, 1), 0)
def accuracy(y_hat, y):
    return (y_hat.argmax(-1) == y)

train_eps = [0.05,0.5,0.75,1]
seeds = [int(sys.argv[1])]
for seed in seeds:
    for eps_iter in range(len(train_eps)):
        print(seed,train_eps[eps_iter])
        model = torch.load('/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BaselineCNN/cel/Jupyter/MTModel/model_Eps_%d_Seed_%d.pt'%(int(train_eps[eps_iter]*1000),seed))
        model.cuda()
        corruption_accuracy = []
        for p in ['gaussian_noise', 'shot_noise', 'motion_blur', 'zoom_blur',
                'spatter', 'brightness' ,'speckle_noise', 'gaussian_blur', 'snow',
                'contrast', 'defocus_blur','elastic_transform','fog','glass_blur','impulse_noise','jpeg_compression',
                'pixelate','saturate','frost']:
            c_p_dir = '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/vision-greg3/CIFAR-10-C'
            norm_layer = trn.Normalize((0.4914, 0.4822, 0.4465), (3*0.2023, 3*0.1994, 3*0.2010))
            dataset = np.load(os.path.join(c_p_dir, p + '.npy'))
            print(np.max(dataset),np.min(dataset),p,dataset.shape)
            dataset = torch.from_numpy(np.transpose(dataset,(0,3,1,2)))/255.
            ood_data = torch.utils.data.TensorDataset(dataset, labels)
            mbs = 100
            loader = torch.utils.data.DataLoader(
                ood_data, batch_size=mbs, shuffle=False, num_workers=2, pin_memory=True)
            
            
            avg_acc = []
            target_list = []
            for data, target in loader:
                data, target = data.cuda(), target.cuda()
                y_hat = model(norm_layer(data))
                acc_val = accuracy(y_hat, target).cpu().numpy()
                avg_acc.append(acc_val)
                target_list.append(target.cpu().numpy())
            avg_acc = np.asarray(avg_acc)
            avg_acc = np.reshape(avg_acc,(5,(avg_acc.shape[1]*avg_acc.shape[0])//5))    
            target_list = np.asarray(target_list)
            
            avg_acc = avg_acc.mean(axis=1)
            
            target_list = np.reshape(target_list,(5,(target_list.shape[0]*target_list.shape[1])//5))
            target_list = target_list.mean(axis=0)
            print(np.sum(isinteger(target_list)))
            corruption_accuracy.append(avg_acc)
        corruption_accuracy = np.asarray(corruption_accuracy)
        np.savetxt('/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BaselineCNN/cel/Jupyter/MTModel/Corruption_Accuracy_Eps_%d_Seed_%d.txt'%(int(train_eps[eps_iter]*1000),seed),
                corruption_accuracy,fmt='%.6f')
        