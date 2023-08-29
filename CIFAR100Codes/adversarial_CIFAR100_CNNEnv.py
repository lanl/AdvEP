import os
from random import randint
import sys
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
from art.estimators.classification import BlackBoxClassifierNeuralNetwork
from art.estimators.classification import BlackBoxClassifier
from art.estimators.estimator import BaseEstimator
from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.attacks.evasion.square_attack import SquareAttack
from art.utils import to_categorical
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision.datasets import CIFAR10, CIFAR100
import matplotlib.cm as cmx
import matplotlib.colors as mcol
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.weight": "bold"
})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Resize, RandomResizedCrop, RandomCrop, Normalize
device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')
    
BATCH_SIZE = 200
EPOCHS = 45
start_time = time.time()
def accuracy(y_hat, y):
    return (y_hat.argmax(-1) == y).float().mean().item()
def train_epoch(train_loader, model, optimizer, loss_fn, mean, std):
    avg_acc = []
    avg_loss = []
    norm_layer = Normalize(mean, std)
    model.train()
    for x, y in train_loader:
        y = y.cuda()
        y_hat = model(norm_layer(x))
        avg_acc.append(accuracy(y_hat, y))
        loss = loss_fn(y_hat, y)
        avg_loss.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()

    return model, sum(avg_acc) / len(avg_acc), sum(avg_loss) / len(avg_loss)
def adv_train_epoch(train_loader, model, optimizer, loss_fn, train_eps, mean, std, norm):
    avg_acc = []
    avg_loss = []
    norm_layer = Normalize(mean, std)
    print(mean,std)
    art_model = PyTorchClassifier(
        model,
        loss_fn,
        (3, 32, 32),
        100,
        optimizer,
        preprocessing=(
            np.array(mean)[None, :, None, None],
            np.array(std)[None, :, None, None]
        )
    )
    attack = ProjectedGradientDescentPyTorch(
        art_model,
        norm,
        train_eps,
        2.5 * train_eps / 20,
        max_iter=20,
        batch_size=train_loader.batch_size,
        verbose=False
    )
    for x, y in train_loader:
        x, y = x, y.cuda()
        model.eval()
        x_adv = attack.generate(x.cpu().numpy())
        model.train()
        model.zero_grad()
        y_hat = model(norm_layer(torch.from_numpy(x_adv)))
        #y_hat = torch.from_numpy(art_model.predict(x_adv)).cuda()
        avg_acc.append(accuracy(y_hat, y))
        loss = loss_fn(y_hat, y)
        avg_loss.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()

    return model, sum(avg_acc) / len(avg_acc), sum(avg_loss) / len(avg_loss)
def adv_val_epoch(train_loader, model, optimizer, loss_fn,mean, std, test_eps,norm):
    cifar_adv, labels, confidence = [], [], []
    avg_acc = []
    avg_loss = []
    norm_layer = Normalize(mean, std)
    art_model = PyTorchClassifier(
        model,
        loss_fn,
        (3, 32, 32),
        100,
        optimizer,
        preprocessing=(
            np.array(mean)[None, :, None, None],
            np.array(std)[None, :, None, None]
        )
    )
    attack = ProjectedGradientDescentPyTorch(
        art_model,
        norm,
        test_eps,
        2.5 * test_eps / 20,
        max_iter=20,
        batch_size=train_loader.batch_size,
        verbose=False
    )
    mbs = train_loader.batch_size
    iter_per_epochs = np.ceil(len(train_loader.dataset)/mbs)
    start = time.time()
    for idx, (x, y) in enumerate(train_loader):
        x, y = x, y.cuda()
        model.eval()
        #print(model(norm_layer(x)).shape)
        #assert all(x[0].size(0) == tensor.size(0) for tensor in x), "Size mismatch between tensors"
        x_adv = attack.generate(x.cpu().numpy())
        cifar_adv.append(x_adv)
        #y_hat = model(norm_layer(torch.from_numpy(x_adv)))
        y_hat = torch.from_numpy(art_model.predict(x_adv)).cuda()
        confidence.append(y_hat.detach().cpu().numpy())
        labels.append(y_hat.argmax(-1).cpu().numpy())  
        avg_acc.append(accuracy(y_hat, y))
        if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)):
                print(
                f'Test Epsilon: {test_eps}; ',
                f'Image Idx: {idx}; ',
                f'Time Elapsed: {time.time()-start}',
                f'Accuracy: {sum(avg_acc) / len(avg_acc)}'
                ) 
        loss = loss_fn(y_hat, y)
        avg_loss.append(loss.item())
    cifar_adv = np.reshape(np.asarray(cifar_adv),(-1,3,32,32))
    labels = np.asarray(labels).flatten()
    confidence = np.reshape(np.asarray(confidence),(-1,100))
    
    return sum(avg_acc) / len(avg_acc), sum(avg_loss) / len(avg_loss),cifar_adv,labels, confidence
def adv_hsj_val_epoch(loader, model, optimizer, loss_fn,mean, std, max_eval,eps, norm):
    cifar_adv, labels, confidence = [], [], []
    avg_acc = []
    avg_loss = []
    avg_predictions = []
    global predict_counter
    predict_counter=0
    
    
    def predict_square(x):
        global predict_counter 
        predict_counter +=1
        y_pred = model(torch.from_numpy(x)).detach().cpu().numpy()
        return y_pred
    
    base_model = BlackBoxClassifierNeuralNetwork(predict_square, input_shape=(3, 32, 32), nb_classes=100,clip_values=(0,1),preprocessing=(
            np.array(mean)[None, :, None, None],
            np.array(std)[None, :, None, None]
        ))
   
    #attack = HopSkipJump(classifier = art_model, norm = norm, targeted = False, max_eval = 10000, max_iter=10, batch_size=loader.batch_size, verbose=True)   
    attack = SquareAttack(base_model, norm = norm,nb_restarts=1, eps = eps,max_iter=max_eval, batch_size=loader.batch_size, verbose=False)   
    mbs = loader.batch_size
    iter_per_epochs = np.ceil(len(loader.dataset)/mbs)
    start = time.time()
    for idx, (x, y) in enumerate(loader):
        x, y = x, y.cuda()
        predict_counter = 0
        model.eval()
        x_adv = None
        for i in range(1):
            x_adv = attack.generate(x.cpu().numpy())
            y_hat = torch.from_numpy(base_model.predict(x_adv)).cuda()
            y_pred = torch.from_numpy(base_model.predict(x.cpu().numpy())).cuda()
            if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)):
                print(
                f'Max_Evals: {max_eval*i}; ',
                f'Image Idx: {idx}; ',
                f'Time Elapsed: {time.time()-start}',
                f'Adv Accuracy: {accuracy(y_hat, y)}',
                f'Clean Accuracy: {accuracy(y_pred, y)}',
                f'Predict Counter: {predict_counter}'
                ) 
            avg_predictions.append(predict_counter)
        cifar_adv.append(x_adv)
        y_hat = torch.from_numpy(base_model.predict(x_adv)).cuda()
        y_pred = torch.from_numpy(base_model.predict(x.cpu().numpy())).cuda()
        confidence.append(y_hat.detach().cpu().numpy())
        labels.append(y_hat.argmax(-1).cpu().numpy())  
        avg_acc.append(accuracy(y_hat, y))
        orig_acc = accuracy(y_pred, y)
        if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)):
                print(
                f'Max_Evals: {max_eval}; ',
                f'Image Idx: {idx}; ',
                f'Time Elapsed: {time.time()-start}',
                f'Adv Accuracy: {sum(avg_acc) / len(avg_acc)}',
                f'Clean Accuracy: {orig_acc}',
                f'Predict Counter: {sum(avg_predictions)/len(avg_predictions)}'
                
                ) 
        loss = loss_fn(y_hat, y)
        avg_loss.append(loss.item())
        break
    cifar_adv = np.reshape(np.asarray(cifar_adv),(-1,3,32,32))
    labels = np.asarray(labels).flatten()
    confidence = np.reshape(np.asarray(confidence),(-1,100))
    
    return sum(avg_acc) / len(avg_acc), sum(avg_loss) / len(avg_loss),sum(avg_predictions)/len(avg_predictions),cifar_adv,labels, confidence
def adv_img_gen(train_loader, model, optimizer, loss_fn, test_eps,train_eps):
    cifar_adv, labels, confidence = [], [], []
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
    mbs = train_loader.batch_size
    iter_per_epochs = np.ceil(len(train_loader.dataset)/mbs)
    start = time.time()
    for idx, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        model.eval()
        #print(model(norm_layer(x)).shape)
        #assert all(x[0].size(0) == tensor.size(0) for tensor in x), "Size mismatch between tensors"
        x_adv = attack.generate(x.cpu().numpy())
        cifar_adv.append(x_adv)
        y_hat = model(norm_layer(torch.from_numpy(x_adv).cuda()))
        model.zero_grad()
        confidence.append(y_hat.detach().cpu().numpy())
        labels.append(y_hat.argmax(-1).cpu().numpy())  
        if ((idx%(iter_per_epochs//10)==0) or (idx==iter_per_epochs-1)):
                print(
                f'Train Epsilon: {train_eps}; ',
                f'Test Epsilon: {test_eps}; ',
                f'Image Idx: {idx}; ',
                f'Time Elapsed: {time.time()-start}'
                ) 
    del art_model         
    cifar_adv = np.reshape(np.asarray(cifar_adv),(-1,3,32,32))
    labels = np.asarray(labels).flatten()
    confidence = np.reshape(np.asarray(confidence),(-1,10))
    return cifar_adv,labels, confidence
def val_epoch(val_loader, model, loss_fn, mean, std):
    avg_acc = []
    avg_loss = []
    norm_layer = Normalize(mean , std)
    model.eval()
    for x, y in val_loader:
        x = x.cuda()
        y = y.cuda()
        y_hat = model(norm_layer(x))
        
        avg_acc.append(accuracy(y_hat, y))
        loss = loss_fn(y_hat, y)
        avg_loss.append(loss.item())
        #print(y_hat.shape,y.shape,avg_acc[-1])
    return sum(avg_acc) / len(avg_acc), sum(avg_loss) / len(avg_loss)
def train(model, train_loader, val_loader, optimizer, loss_fn, scheduler, n_epochs, mean, std, norm = 2, adv_training=False, train_eps= 0.001):
    for epoch in range(1, n_epochs + 1):
        if adv_training:
            model, train_acc, train_loss = adv_train_epoch(train_loader, model, optimizer, loss_fn, train_eps, mean, std, norm)
            val_acc, val_loss = val_epoch(val_loader, model, loss_fn, mean, std)
        else:
            model, train_acc, train_loss = train_epoch(train_loader, model, optimizer, loss_fn, mean, std)
            val_acc, val_loss = val_epoch(val_loader, model, loss_fn, mean, std)
            
        scheduler.step()
        print(
            f'Epoch: {epoch}; ',
            f'Train Acc: {round(train_acc, 2)}; ',
            f'Train Loss: {round(train_loss, 4)}; ',
            f'Val Acc: {round(val_acc, 2)}; ',
            f'Val Loss: {round(val_loss, 4)} ',
            f'LR: {scheduler.get_last_lr()[0]:.2e} '
            f'Time Elapsed: {time.time()-start_time}'
        )
    return model
def adv_test(model, train_loader, val_loader, optimizer, loss_fn, mean, std, scheduler, n_epochs, adv_training=False, train_eps = 0.001, seed=0,norm=2,model_str=''):
    if(norm==100):
        test_eps = np.logspace(np.log10(0.0001),np.log10(0.1),10)  
        test_eps = np.insert(test_eps,0,1)
        test_eps = np.insert(test_eps,0,8/255)
        test_eps = np.sort(test_eps)
    else:

        test_eps = np.logspace(np.log10(0.05),np.log10(3),10)
        test_eps = np.insert(test_eps,0,1)
        test_eps = np.insert(test_eps,0,0.005)
        test_eps = np.sort(test_eps)
    accuracy_values = np.zeros((len(test_eps),3))
    norm_str = norm
    if(norm==100):
        norm=np.inf
    load_path = '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/CIFAR100/results/MTModel/'
    for i in range(len(test_eps)):
        train_acc, train_loss = val_epoch(train_loader, model, loss_fn,mean, std)
        val_acc, val_loss,cifar_adv, labels, confidence = adv_val_epoch(val_loader, model, optimizer, loss_fn, mean, std, test_eps[i],norm)
        accuracy_values[i] = test_eps[i], train_acc, val_acc
        if(train_eps==0):
            np.save(load_path + 'Adversarial_Norm_%d_Images_Model_%s_Test_Eps_%d_Seed_%d.npy'%(norm_str,model_str,int(test_eps[i]*1000),seed),
                cifar_adv)
            np.save(load_path + 'Labels_Norm_%d_Model_%s_Test_Eps_%d_Seed_%d.npy'%(norm_str,model_str,int(test_eps[i]*1000),seed),
                labels)
            np.save(load_path + 'Confidence_Norm_%d_Model_%s_Test_Eps_%d_Seed_%d.npy'%(norm_str,model_str,int(test_eps[i]*1000),seed),
                confidence)
            np.savetxt(load_path+'Adversarial_Norm_%d_Accuracy_Model_%s_Seed_%d.txt'%(norm_str,model_str,seed),
                    accuracy_values,fmt='%.6f')
        else:
            np.save(load_path + 'Adversarial_Norm_%d_Images_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'%(norm_str,int(train_eps*1000),int(test_eps[i]*1000),seed),
                cifar_adv)
            np.save(load_path + 'Labels_Norm_%d_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'%(norm_str,int(train_eps*1000),int(test_eps[i]*1000),seed),
                labels)
            np.save(load_path + 'Confidence_Norm_%d_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'%(norm_str,int(train_eps*1000),int(test_eps[i]*1000),seed),
                confidence)
            np.savetxt(load_path+'Adversarial_Norm_%d_Accuracy_Eps_%d_Seed_%d.txt'%(norm_str,int(train_eps*1000),seed),
                    accuracy_values,fmt='%.6f')
        print(
            f'Epsilon: {test_eps[i]}; ',
            f'Train Acc: {round(train_acc, 2)}; ',
            f'Train Loss: {round(train_loss, 4)}; ',
            f'Val Acc: {round(val_acc, 2)}; ',
            f'Val Loss: {round(val_loss, 4)} ',
            f'Time Elapsed: {time.time()-start_time}'
        )
        
    return model
def adv_hsj_test(model, train_loader, val_loader, optimizer, loss_fn, mean, std,  scheduler, n_epochs, adv_training=False, train_eps = 0.001, seed=0,norm=2,model_str=''):
    max_evals = [1000]
    if(norm==100):
        test_eps = np.logspace(np.log10(0.0001),np.log10(0.1),10)  
        test_eps = np.insert(test_eps,0,8/255)
        test_eps = np.sort(test_eps)
    else:

        test_eps = np.logspace(np.log10(0.05),np.log10(3),10)
        test_eps = np.insert(test_eps,0,1)
        test_eps = np.insert(test_eps,0,0.005)
        test_eps = np.sort(test_eps)
    norm_str = norm
    if(norm==100):
        norm=np.inf
    load_path = '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/CIFAR100/results/MTModel/'
    for i in range(len(max_evals)):
        accuracy_values = np.zeros((len(test_eps),4))

        for j in range(3,len(test_eps)):
        
            train_acc, train_loss = val_epoch(train_loader, model, loss_fn, mean, std)
            val_acc, val_loss, avg_predictions, cifar_adv, labels, confidence = adv_hsj_val_epoch(val_loader, model, optimizer, loss_fn, mean, std, max_evals[i],test_eps[j],norm)
            accuracy_values[j] = test_eps[j], train_acc, val_acc, avg_predictions
            if(train_eps==0):
                np.save(load_path + 'Adversarial_HSJ_Norm_%d_Images_Model_%s_Max_Evals_%d_Test_Eps_%d_Seed_%d.npy'%(norm_str,model_str,int(max_evals[i]*1000),int(test_eps[j]*1000),seed),
                    cifar_adv)
                np.save(load_path + 'Labels_HSJ_Norm_%d_Model_%s_Max_Evals_%d_Test_Eps_%d_Seed_%d.npy'%(norm_str,model_str,int(max_evals[i]*1000),int(test_eps[j]*1000),seed),
                    labels)
                np.save(load_path + 'Confidence_HSJ_Norm_%d_Model_%s_Max_Evals_%d_Test_Eps_%d_Seed_%d.npy'%(norm_str,model_str,int(max_evals[i]*1000),int(test_eps[j]*1000),seed),
                    confidence)
                np.savetxt(load_path+'Adversarial_HSJ_Norm_%d_Accuracy_Model_%s_Max_Evals_%d_Seed_%d.txt'%(norm_str,model_str,int(max_evals[i]*1000),seed),
                        accuracy_values,fmt='%.6f')
            else:
                np.save(load_path + 'Adversarial_HSJ_Norm_%d_Images_Train_Eps_%d_Max_Evals_%d_Test_Eps_%d_Seed_%d.npy'%(norm_str,int(train_eps*1000),int(max_evals[i]*1000),int(test_eps[j]*1000),seed),
                    cifar_adv)
                np.save(load_path + 'Labels_HSJ_Norm_%d_Train_Eps_%d_Max_Evals_%d_Test_Eps_%d_Seed_%d.npy'%(norm_str,int(train_eps*1000),int(max_evals[i]*1000),int(test_eps[j]*1000),seed),
                    labels)
                np.save(load_path + 'Confidence_HSJ_Norm_%d_Train_Eps_%d_Max_Evals_%d_Test_Eps_%d_Seed_%d.npy'%(norm_str,int(train_eps*1000),int(max_evals[i]*1000),int(test_eps[j]*1000),seed),
                    confidence)
                np.savetxt(load_path+'Adversarial_HSJ_Norm_%d_Accuracy_Eps_%d_Max_Evals_%d_Seed_%d.txt'%(norm_str,int(train_eps*1000),int(max_evals[i]*1000),seed),
                        accuracy_values,fmt='%.6f')
            print(
                f'Max_Evals: {max_evals[i]}; ',
                f'Train Acc: {round(train_acc, 2)}; ',
                f'Train Loss: {round(train_loss, 4)}; ',
                f'Val Acc: {round(val_acc, 2)}; ',
                f'Val Loss: {round(val_loss, 4)} ',
                f'Time Elapsed: {time.time()-start_time}'
            )
        
    return model
def adv_generate(model, train_loader, val_loader, optimizer, loss_fn, scheduler, n_epochs, adv_training=False, train_eps = 0.001, seed=0):
    test_eps = np.logspace(np.log10(0.05),np.log10(3),10)
    test_eps = np.insert(test_eps,0,1)
    test_eps = np.sort(test_eps)
    test_eps = [1]
    cifar_adv, labels = [], []
    load_path = '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BaselineCNN/cel/Jupyter/MTModel/'
    for i in range(len(test_eps)):
                
        model.zero_grad()
        cifar_adv, labels, confidence  = adv_img_gen(val_loader, model, optimizer, loss_fn, test_eps[i],train_eps)
        np.save(load_path + 'Adversarial_Images_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'%(int(train_eps*1000),int(test_eps[i]*1000),seed),
            cifar_adv)
        np.save(load_path + 'Labels_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'%(int(train_eps*1000),int(test_eps[i]*1000),seed),
            labels)
        np.save(load_path + 'Confidence_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'%(int(train_eps*1000),int(test_eps[i]*1000),seed),
            confidence)
        print(
            f'Epsilon: {test_eps[i]}; ',
            f'i: {i}; ',
    )
    
def dataload():
    train_data = CIFAR100('./cifar100_pytorch', train=True, download=True)

    # Stick all the images together to form a 1600000 X 32 X 3 array
    x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])

    # calculate the mean and std along the (0, 1) axes
    mean = np.mean(x, axis=(0, 1))/255
    std = np.std(x, axis=(0, 1))/255
    # the the mean and std
    mean=mean.tolist()
    std = std*3
    std=std.tolist()
    print(mean,std)
    train_dset = CIFAR100(
        './data/cifar',
        train=True,
        transform=Compose([
            RandomCrop(32, padding=4,padding_mode='reflect'),
            RandomHorizontalFlip(),
            ToTensor(),
        ]),
        download=True
    )
    test_dset = CIFAR100(
        './data/cifar',
        train=False,
        transform=Compose([
            ToTensor()
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
    return train_loader, test_loader, mean, std

seed = int(sys.argv[1])
adv_train = int(sys.argv[2])
#random.seed(seed)
#np.random.seed(seed)
def adv_seed_train(seed, adv_train=False):
    norm_str = 100
    norm = np.inf
    torch.manual_seed(seed)
    train_loader, test_loader, mean, std = dataload()
    if adv_train ==False:
        train_eps = [0]
    else:
        if norm==2:
            train_eps = [0.05,0.5,0.75,1]
        else:
            train_eps = [8.0/255]
    for i in range(len(train_eps)):
        print(seed, train_eps[i])
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
        nn.Conv2d(512, 1024, 3, 1, 0, bias=False),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(1024, 100),
        #nn.Unflatten(0,(1,200)),
    )
        model = nn.DataParallel(model, [0]).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1,momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()
        scheduler = StepLR(optimizer, 8, 0.3)
        print(model)
        
        model = train(
            model, 
            train_loader, 
            test_loader, 
            optimizer, 
            loss_fn, 
            scheduler, 
            EPOCHS, mean, std, norm,
            adv_train, train_eps[i]
        )
        if(adv_train==0):
            torch.save(model,'/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/CIFAR100/results/MTModel/model_relu_Seed_%d.pt'%(seed))
        else:
            torch.save(model,'/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/CIFAR100/results/MTModel/norm_%d_model_Eps_%d_Seed_%d.pt'%(norm_str,int(train_eps[i]*1000),seed))

def adv_seed_test(seed):
    train_eps = [0.05,0.5,0.75,1]
    for i in range(len(train_eps)):
        model = torch.load('/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/CIFAR100/results/MTModel/model_Eps_%d_Seed_%d.pt'%(int(train_eps[i]*1000),seed))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1,momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()
        train_loader, test_loader, mean ,std = dataload()
        scheduler = StepLR(optimizer, 8, 0.3)
        norm = int(sys.argv[2])
        model = adv_test(
            model, 
            train_loader, 
            test_loader, 
            optimizer, 
            loss_fn, 
            mean ,std, 
            scheduler, 
            EPOCHS, 
            True,train_eps[i],seed,norm)
def adv_hsj_seed_test(seed):
    train_eps = [0.05,0.5,0.75,1]
    for i in range(len(train_eps)):
        model = torch.load('/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/CIFAR100/results/MTModel/model_Eps_%d_Seed_%d.pt'%(int(train_eps[i]*1000),seed))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1,momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()
        train_loader, test_loader, mean, std = dataload()
        scheduler = StepLR(optimizer, 8, 0.3)
        norm = int(sys.argv[2])
        model = adv_hsj_test(
            model, 
            train_loader, 
            test_loader, 
            optimizer, 
            loss_fn, 
            mean, std, 
            scheduler, 
            EPOCHS, 
            True,train_eps[i],seed,norm)
def adv_seed_test_relu_hs(seed):
    model_str = ['hardsigmoid','relu']
    for i in range(len(model_str)):
        model = torch.load('/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BaselineCNN/cel/Jupyter/MTModel/model_%s.pt'%(model_str[i]))
        optimizer = torch.optim.Adam(model.parameters(), 3e-3)
        optimizer = torch.optim.SGD(model.parameters(), 0.9)
        loss_fn = nn.CrossEntropyLoss()
        train_loader, test_loader = dataload()
        scheduler = StepLR(optimizer, 8, 0.3)
        norm = int(sys.argv[2])
        model = adv_test(
            model, 
            train_loader, 
            test_loader, 
            optimizer, 
            loss_fn, 
            scheduler, 
            EPOCHS, 
            True,0,seed,norm,model_str[i])
def adv_hsj_seed_test_relu_hs(seed):
    model_str = ['hardsigmoid','relu']
    for i in range(1,len(model_str)):
        model = torch.load('/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/CIFAR100/results/MTModel/model_%s_Seed_%d.pt'%(model_str[i],seed))
        optimizer = torch.optim.Adam(model.parameters(), 3e-3)
        optimizer = torch.optim.SGD(model.parameters(), 0.9)
        loss_fn = nn.CrossEntropyLoss()
        train_loader, test_loader, mean, std = dataload()
        scheduler = StepLR(optimizer, 8, 0.3)
        norm = int(sys.argv[2])
        model = adv_hsj_test(
            model, 
            train_loader, 
            test_loader, 
            optimizer, 
            loss_fn, 
            mean, std, 
            scheduler, 
            EPOCHS, 
            True,0,seed,norm,model_str[i])
def adv_seed_gen(seed):
    #train_eps = [0.05,0.5,0.75,1]
    train_eps = [1]
    for i in range(len(train_eps)):
        model = torch.load('/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BaselineCNN/cel/Jupyter/MTModel/model_Eps_%d_Seed_%d.pt'%(int(train_eps[i]*1000),seed))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), 3e-3)
        loss_fn = nn.CrossEntropyLoss()
        train_loader, test_loader = dataload()
        scheduler = StepLR(optimizer, 8, 0.3)
        adv_generate(
            model, 
            train_loader, 
            test_loader, 
            optimizer, 
            loss_fn, 
            scheduler, 
            EPOCHS, 
            True,train_eps[i],seed)
def adv_img_histogram(seed,test_eps):
    train_eps = [0.05,0.5,0.75,1]
    """ test_eps = np.logspace(np.log10(0.05),np.log10(3),10)
    test_eps = np.insert(test_eps,0,1)
    test_eps = np.sort(test_eps) """
    test_eps = [test_eps]
    load_path = '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BaselineCNN/cel/Jupyter/MTModel/'
    filename = 'Adversarial_Images_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'%(int(train_eps[0]*1000),int(test_eps[0]*1000),seed)
    labels_filename = 'Labels_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'%(int(train_eps[0]*1000),int(test_eps[0]*1000),seed)
    labels = torch.from_numpy(np.load(os.path.join(load_path, labels_filename)))
    dataset = torch.from_numpy(np.load(os.path.join(load_path, filename )))
    diff = []
    for i in range(1,len(train_eps)):
        filename = 'Adversarial_Images_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'%(int(train_eps[i]*1000),int(test_eps[0]*1000),seed)
        labels_filename = 'Labels_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'%(int(train_eps[i]*1000),int(test_eps[0]*1000),seed)
        compare_labels = torch.from_numpy(np.load(os.path.join(load_path, labels_filename)))
        compare_dataset = torch.from_numpy(np.load(os.path.join(load_path, filename )))
        multiple = 30

        diff.append((compare_dataset[:multiple*BATCH_SIZE]-dataset[:multiple*BATCH_SIZE])**2)
    return diff
def adv_img_histogram_EP(seed,diff,test_eps):
    train_eps = [0.05,0.5,0.75,1]
    """ test_eps = np.logspace(np.log10(0.05),np.log10(3),10)
    test_eps = np.insert(test_eps,0,1)
    test_eps = np.sort(test_eps) """
    test_eps = [test_eps]
    attack_t = 29
    predict_t = 250
    load_path = '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BaselineCNN/cel/Jupyter/MTModel/'
    filename = 'Adversarial_Images_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'%(int(train_eps[0]*1000),int(test_eps[0]*1000),seed)
    labels_filename = 'Labels_Train_Eps_%d_Test_Eps_%d_Seed_%d.npy'%(int(train_eps[0]*1000),int(test_eps[0]*1000),seed)
    labels = torch.from_numpy(np.load(os.path.join(load_path, labels_filename)))
    dataset = torch.from_numpy(np.load(os.path.join(load_path, filename )))
    test_path = '/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/cel/2023-06-14/17-04-29_gpu0/'
    for i in range(1,len(train_eps)):
        filename = 'White_Last_Layer_Adversarial_Images_Eps_%d_Attack_T_%d_Predict_T_%d.npy'%(int(test_eps[0]*1000),attack_t,predict_t)
        labels_filename = 'White_Last_Layer_Labels_Eps_%d_Attack_T_%d_Predict_T_%d.npy'%(int(test_eps[0]*1000),attack_t,predict_t)
        compare_labels = torch.from_numpy(np.load(os.path.join(test_path, labels_filename)))
        compare_dataset = torch.from_numpy(np.load(os.path.join(test_path, filename )))
        multiple = 30
        print(compare_dataset.shape,dataset.shape)
        diff.append((compare_dataset[:multiple*BATCH_SIZE]-dataset[:multiple*BATCH_SIZE])**2)
        print(labels[0],compare_labels[0])
        break
    return diff
def plot_histogram(diff,ax):
    train_eps = [0.5,0.75,1]
    cm = plt.get_cmap('jet')
    cm = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
    cNorm = mcol.Normalize(vmin=min(train_eps), vmax=max(train_eps))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    

    for i in range(len(diff)-1):
        binmin, binmax = torch.min(torch.abs(diff[i])),torch.max(torch.abs(diff[i]))
        bins = np.logspace(np.log10(10**-6),np.log10(binmax),20)
        hist, bin_edges = np.histogram(np.abs(diff[i]), bins = bins,density=True)
        ax.plot(bin_edges[:-1],hist,c=scalarMap.to_rgba(train_eps[i]),label=r'VCNN $(\epsilon=%.2f)$'%train_eps[i])
    for i in range(len(diff)-1,len(diff)):
        binmin, binmax = torch.min(torch.abs(diff[i])),torch.max(torch.abs(diff[i]))
        bins = np.logspace(np.log10(10**-6),np.log10(binmax),20)
        hist, bin_edges = np.histogram(np.abs(diff[i]), bins = bins,density=True)
        ax.plot(bin_edges[:-1],hist,c='k',label='EP Symmetric Gradient: Loss CE')
    
#adv_seed_test_relu_hs(seed)
adv_seed_train(seed,adv_train)
#adv_seed_test(seed)
#adv_hsj_seed_test(seed)
#adv_hsj_seed_test_relu_hs(seed)
def cross_evaluate(load_path,EP_seed,CNN_seed):
    test_eps = np.logspace(np.log10(0.1),np.log10(3),20)
    #print(test_eps)
    test_eps= [0.5,1]
    train_eps = [0.05,0.5,0.75,1]
    accuracy_values = np.zeros((len(train_eps),len(test_eps)))
    for i in range(len(train_eps)):
        model = torch.load('/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BaselineCNN/cel/Jupyter/MTModel/model_Eps_%d_Seed_%d.pt'%(int(train_eps[i]*1000),CNN_seed))
        model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        for j in range(len(test_eps)):
            filename = 'Adversarial_Images_Eps_%d_2'%(int(test_eps[j]*1000))
            
            labels = torch.from_numpy(np.load(os.path.join(load_path, 'Labels_2.npy')))
            dataset = torch.from_numpy(np.load(os.path.join(load_path, filename + '.npy')))
            
            ood_data = torch.utils.data.TensorDataset(dataset, labels)
            mbs = 100
            loader = torch.utils.data.DataLoader(
                ood_data, batch_size=mbs, shuffle=False, num_workers=2, pin_memory=True)
            accuracy_values[i][j],_ = val_epoch(loader, model, loss_fn)
            np.savetxt('/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/BaselineCNN/cel/Jupyter/MTModel/CrossValidation_Accuracy_EPSeed_%d_CNNSeed_%d.txt'%(EP_seed,CNN_seed),accuracy_values,fmt='%.6f')
load_path_seed = [['/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/cel/2023-06-14/17-04-29_gpu0/',11],
                  ['/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/cel/2023-06-14/17-06-02_gpu0/',13],
                  ['/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/cel/2023-06-14/17-06-08_gpu0/',17],
                  ['/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/cel/2023-06-14/17-06-24_gpu0/',19],
                  ['/vast/home/smansingh/LaborieuxEP/Equilibrium-Propagation/results/EP/cel/2023-06-15/00-48-27_gpu0/',31]]
""" for i in range(len(load_path_seed)):
    cross_evaluate(load_path_seed[i][0],load_path_seed[i][1],seed)
    break """
#adv_seed_gen(seed)
def histogram_perturbations():
    test_eps = np.logspace(np.log10(0.05),np.log10(3),10)
    test_eps = np.insert(test_eps,0,1)
    test_eps = np.sort(test_eps) 

    fig,ax = plt.subplots(1,1,figsize=(10,8))
    diff = adv_img_histogram(seed,test_eps[4])
    diff = adv_img_histogram_EP(seed,diff,test_eps[4])
    plot_histogram(diff,ax)
    ax.text(2*10**-4,10**3,r'Test $\epsilon=%.2f$'%test_eps[4],fontsize=20)
    diff = adv_img_histogram(seed,test_eps[7])
    diff = adv_img_histogram_EP(seed,diff,test_eps[7])
    plot_histogram(diff,ax)
    ax.text(5*10**-3,10**2,r'Test $\epsilon=%.2f$'%test_eps[7],fontsize=20)
    diff = adv_img_histogram(seed,test_eps[10])
    diff = adv_img_histogram_EP(seed,diff,test_eps[10])
    plot_histogram(diff,ax)
    ax.text(7*10**-2,10**1,r'Test $\epsilon=%.2f$'%test_eps[10],fontsize=20)
    #ax.legend(loc=3,fontsize=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.set_xlim([10**-3,0.2])
    ax.set_xlabel(r'Perturbation $\delta$',fontsize=35)
    ax.set_ylabel(r'$P(\delta)$',fontsize=35)
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    
    plt.tight_layout()
    fig.savefig('Histogram_Perturbations.png')
#histogram_perturbations()