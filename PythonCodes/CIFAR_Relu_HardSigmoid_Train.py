import os
from random import randint

from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier


import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision.datasets import CIFAR10

from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Resize, RandomResizedCrop, RandomCrop, Normalize

BATCH_SIZE = 200
EPOCHS = 45

def accuracy(y_hat, y):
    return (y_hat.argmax(-1) == y).float().mean().item()


def train_epoch(train_loader, model, optimizer, loss_fn):
    avg_acc = []
    avg_loss = []
    norm_layer = Normalize((0.4914, 0.4822, 0.4465), (3*0.2023, 3*0.1994, 3*0.2010))
    model.train()
    for x, y in train_loader:
        y = y.cuda()
        y_hat = model(x)
        avg_acc.append(accuracy(y_hat, y))
        loss = loss_fn(y_hat, y)
        avg_loss.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()

    return model, sum(avg_acc) / len(avg_acc), sum(avg_loss) / len(avg_loss)
def adv_val_epoch(train_loader, model, optimizer, loss_fn, test_eps):
    avg_acc = []
    avg_loss = []
    norm_layer = Normalize((0.4914, 0.4822, 0.4465), (3*0.2023, 3*0.1994, 3*0.2010))
    art_model = PyTorchClassifier(
        model,
        loss_fn,
        (3, 32, 32),
        10,
        optimizer,
        preprocessing=(
            np.array([0.4914, 0.4822, 0.4465])[None, :, None, None],
            np.array([3*0.2023, 3*0.1994, 3*0.2010])[None, :, None, None]
        )
    )
    attack = ProjectedGradientDescentPyTorch(
        art_model,
        2,
        test_eps,
        2.5 * test_eps / 20,
        max_iter=20,
        batch_size=train_loader.batch_size,
        verbose=False
    )
    for x, y in train_loader:
        x, y = x, y.cuda()
        model.eval()
        x_adv = attack.generate(x.cpu().numpy())
        y_hat = model(norm_layer(torch.from_numpy(x_adv)))
        avg_acc.append(accuracy(y_hat, y))
        loss = loss_fn(y_hat, y)
        avg_loss.append(loss.item())
    return sum(avg_acc) / len(avg_acc), sum(avg_loss) / len(avg_loss)
def val_epoch(val_loader, model, loss_fn):
    avg_acc = []
    avg_loss = []
    norm_layer = Normalize((0.4914, 0.4822, 0.4465), (3*0.2023, 3*0.1994, 3*0.2010))
    model.eval()
    for x, y in val_loader:
        y = y.cuda()
        y_hat = model(x)
        avg_acc.append(accuracy(y_hat, y))
        loss = loss_fn(y_hat, y)
        avg_loss.append(loss.item())

    return sum(avg_acc) / len(avg_acc), sum(avg_loss) / len(avg_loss)

def train(model, train_loader, val_loader, optimizer, loss_fn, scheduler, n_epochs, adv_training=False, train_eps= 0.001):
    for epoch in range(1, n_epochs + 1):
        if adv_training:
            model, train_acc, train_loss = adv_train_epoch(train_loader, model, optimizer, loss_fn, train_eps)
            val_acc, val_loss = val_epoch(val_loader, model, loss_fn)
        else:
            model, train_acc, train_loss = train_epoch(train_loader, model, optimizer, loss_fn)
            val_acc, val_loss = val_epoch(val_loader, model, loss_fn)
            
        scheduler.step()
        print(
            f'Epoch: {epoch}; ',
            f'Train Acc: {round(train_acc, 2)}; ',
            f'Train Loss: {round(train_loss, 4)}; ',
            f'Val Acc: {round(val_acc, 2)}; ',
            f'Val Loss: {round(val_loss, 4)} ',
            f'LR: {scheduler.get_last_lr()[0]:.2e}'
        )
    return model

def adv_test(model, train_loader, val_loader, optimizer, loss_fn, scheduler, n_epochs, adv_training=False, train_eps = 0.001):
    test_eps = np.logspace(np.log10(0.05),np.log10(3),20)
    #test_eps = [0.01,1]
    accuracy_values = np.zeros((len(test_eps),3))
    for i in range(len(test_eps)):
        train_acc, train_loss = val_epoch(train_loader, model, loss_fn)
        val_acc, val_loss = adv_val_epoch(val_loader, model, optimizer, loss_fn, test_eps[i])
        accuracy_values[i] = test_eps[i], train_acc, val_acc
        print(
            f'Epsilon: {test_eps[i]}; ',
            f'Train Acc: {round(train_acc, 2)}; ',
            f'Train Loss: {round(train_loss, 4)}; ',
            f'Val Acc: {round(val_acc, 2)}; ',
            f'Val Loss: {round(val_loss, 4)} ',
    )
    
    return model, accuracy_values

train_dset = CIFAR10(
    './data/cifar',
    train=True,
    transform=Compose([RandomHorizontalFlip(0.5),
                        RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
                        ToTensor(), 
                        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(3*0.2023, 3*0.1994, 3*0.2010))
    ]),
    download=True
    )
test_dset = CIFAR10(
    './data/cifar',
    train=False,
    transform=Compose([
        ToTensor(), Normalize(mean=(0.4914, 0.4822, 0.4465), std=(3*0.2023, 3*0.1994, 3*0.2010)),
    ]),
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
def train_relu():
    model = nn.Sequential(
    nn.Conv2d(3, 128, 3, 1, 1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(128, 256, 3, 1, 1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(256, 512, 3, 1, 1, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(512, 512, 3, 1, 0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(512, 10),
    #nn.Unflatten(0,(1,200)),
    )
    model = nn.DataParallel(model, [0]).cuda()
    #optimizer = torch.optim.Adam(model.parameters(), 3e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, 8, 0.3)
    model = train(
        model, 
        train_loader, 
        test_loader, 
        optimizer, 
        loss_fn, 
        scheduler, 
        EPOCHS, 
        False
    )
    torch.save(model, 
    '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BaselineCNN/cel/Jupyter/MTModel/model_relu.pt')

def train_hardsigmoid():

    model = nn.Sequential(
    nn.Conv2d(3, 128, 3, 1, 1, bias=False),
    nn.BatchNorm2d(128),
    nn.Hardsigmoid(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(128, 256, 3, 1, 1, bias=False),
    nn.BatchNorm2d(256),
    nn.Hardsigmoid(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(256, 512, 3, 1, 1, bias=False),
    nn.BatchNorm2d(512),
    nn.Hardsigmoid(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(512, 512, 3, 1, 0, bias=False),
    nn.BatchNorm2d(512),
    nn.Hardsigmoid(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(512, 10),
    #nn.Unflatten(0,(1,200)),
    )
    model = nn.DataParallel(model, [0]).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(),3e-3)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, 8, 0.3)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-5)
    model = train(
        model, 
        train_loader, 
        test_loader, 
        optimizer, 
        loss_fn, 
        scheduler, 
        EPOCHS, 
        False
    )
    torch.save(model, 
    '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BaselineCNN/cel/Jupyter/MTModel/model_hardsigmoid.pt')
#train_relu()
train_hardsigmoid()