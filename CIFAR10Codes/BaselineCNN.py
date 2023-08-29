import os
from random import randint

from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
#import h5py
#from imageio import imwrite
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torchvision
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision.datasets import CIFAR10
#from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Resize, RandomResizedCrop, Normalize
os.chdir('/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BaselineCNN/cel/Jupyter/')
BATCH_SIZE = 200
EPOCHS = 40
def accuracy(y_hat, y):
    return (y_hat.argmax(-1) == y).float().mean().item()
def train_epoch(train_loader, model, optimizer, loss_fn, scheduler):
    avg_acc = []
    avg_loss = []
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        
        avg_acc.append(accuracy(y_hat[-1], y))
        loss = loss_fn(y_hat[-1], y)
        avg_loss.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return model, sum(avg_acc) / len(avg_acc), sum(avg_loss) / len(avg_loss)

def adv_train_epoch(train_loader, model, optimizer, loss_fn, scheduler):
    avg_acc = []
    avg_loss = []
    mean = np.array([0.4914, 0.4822, 0.4465])[None,:,None,None]
    std=np.array([3*0.2023, 3*0.1994, 3*0.2010])[None,:,None,None]
    eps = 1e-3
    art_model = PyTorchClassifier(
        model,
        loss_fn,
        (3, 32, 32),
        10,
        optimizer, clip_values=None,preprocessing=(mean,std),
    )
    attack = ProjectedGradientDescentPyTorch(
        art_model,
        2,
        eps,
        2.5 * eps / 20,
        max_iter=20,
        batch_size=train_loader.batch_size,
        verbose=False
    )
    #print(model.device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for x, y in train_loader:
        #x, y = x.to(device), y.to(device) 
        #y = y.to(device)
        #art_model.eval()
        #print(x.shape)
        x_adv = attack.generate(x.cpu())
        
        #art_model.train()
        y_hat = art_model.model(torch.from_numpy(x_adv).to(device)).to('cpu')
        
        avg_acc.append(accuracy(y_hat[-1], y))
        #print(y_hat.argmax(-1)[:10],y[:10])
        loss = loss_fn(y_hat[-1], y)
        avg_loss.append(loss.item())
        """ art_model.model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() """

    return art_model.model, sum(avg_acc) / len(avg_acc), sum(avg_loss) / len(avg_loss)
def val_epoch(val_loader, model, loss_fn):
    avg_acc = []
    avg_loss = []
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        avg_acc.append(accuracy(y_hat[-1], y))
        loss = loss_fn(y_hat[-1], y)
        avg_loss.append(loss.item())

    return sum(avg_acc) / len(avg_acc), sum(avg_loss) / len(avg_loss)
def train(model, train_loader, val_loader, optimizer, loss_fn, scheduler, n_epochs, adv_training=False):
    for epoch in range(1, n_epochs + 1):
        if adv_training:
            model, train_acc, train_loss = adv_train_epoch(train_loader, model, optimizer, loss_fn, scheduler)
            val_acc, val_loss = val_epoch(val_loader, model, loss_fn)
        else:
            model, train_acc, train_loss = train_epoch(train_loader, model, optimizer, loss_fn, scheduler)
            val_acc, val_loss = val_epoch(val_loader, model, loss_fn)
        print(
            f'Epoch: {epoch}; ',
            f'Train Acc: {round(train_acc, 2)}; ',
            f'Train Loss: {round(train_loss, 4)}; ',
            f'Val Acc: {round(val_acc, 2)}; ',
            f'Val Loss: {round(val_loss, 4)} ',
            f'LR: {scheduler.get_last_lr()[0]:.2e}'
        )
    return model
###### make sure the transforms below are the same as in the EQProp code you are using ########
def DataLoaderTransforms(attack):
    if(attack==0):
        transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(0.5),
                                                                torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
                                                                torchvision.transforms.ToTensor(), 
                                                                torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                                                                std=(3*0.2023, 3*0.1994, 3*0.2010)) ])


        transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                            torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                                                            std=(3*0.2023, 3*0.1994, 3*0.2010)) ]) 

    else:
        transform_train = torchvision.transforms.ToTensor()

        transform_test = torchvision.transforms.ToTensor()  

    return transform_train, transform_test
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128, affine=True),
                nn.Hardsigmoid(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation = 1, ceil_mode=False),
            
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256, affine=True),
                nn.Hardsigmoid(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation = 1, ceil_mode=False),
                
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512, affine=True),
                nn.Hardsigmoid(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation = 1, ceil_mode=False),
                
                nn.Conv2d(512, 512, kernel_size=3, stride=1),
                nn.BatchNorm2d(512, affine=True),
                nn.Hardsigmoid(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation = 1, ceil_mode=False),
            
                nn.Flatten(),
                nn.Linear(512, 10, bias=True),
                #nn.Hardsigmoid(inplace=True),
                #nn.LogSoftmax(dim=-1)
                )

    def forward(self, x):
        x = self.classifier(x)
        x = torch.unsqueeze(x,0)
        return x
##### Define a model with the same number of layers and filters in the line below #####

#model = nn.DataParallel(model).cuda()


loss_fn = nn.CrossEntropyLoss()
attack = 1
transform_train, transform_test = DataLoaderTransforms(attack)
train_dset = CIFAR10(
            './data/cifar',
            train=True,
            transform=transform_train,
            download=True
        )
test_dset = CIFAR10(
    './data/cifar',
    train=False,
    transform=transform_test,
    download=True
)

train_loader = DataLoader(
    train_dset,
    BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)
test_loader = DataLoader(
    test_dset,
    BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

if(attack==0):
    model = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-4, momentum=0, 
                                dampening=0, weight_decay=0, 
                                nesterov=False)
    scheduler = OneCycleLR(
    optimizer=optimizer,
    max_lr=1,
    epochs=EPOCHS,
    steps_per_epoch=len(train_dset) // BATCH_SIZE,
    div_factor=1e1,
    final_div_factor=1e2,
    three_phase=True
    )
    model.to(device)
    model = train(model, train_loader, test_loader, optimizer, loss_fn, scheduler, EPOCHS)
    torch.save(model, '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BaselineCNN/cel/Jupyter/Conv5_model.pt')
else:
    model = torch.load('/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BaselineCNN/cel/Jupyter/Conv5_model.pt')
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-4, momentum=0, 
                                dampening=0, weight_decay=0, 
                                nesterov=False)
    
    scheduler = OneCycleLR(
    optimizer=optimizer,
    max_lr=1,
    epochs=EPOCHS,
    steps_per_epoch=len(train_dset) // BATCH_SIZE,
    div_factor=1e1,
    final_div_factor=1e2,
    three_phase=True
    )
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print(model)

    art_model = train(model, 
    train_loader, 
    test_loader, 
    optimizer, 
    loss_fn, 
    scheduler, 
    EPOCHS, 
    True)