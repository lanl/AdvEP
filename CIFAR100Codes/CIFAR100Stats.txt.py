elif args.task=='CIFAR100':
    train_data = torchvision.datasets.CIFAR100('./cifar100_pytorch', train=True, download=False)

    # Stick all the images together to form a 1600000 X 32 X 3 array
    x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])

    # calculate the mean and std along the (0, 1) axes
    mean = np.mean(x, axis=(0, 1))/255
    std = np.std(x, axis=(0, 1))/255
    # the the mean and std
    mean=mean.tolist()
    std=std.tolist()
    print(mean,std)
    transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4,padding_mode='reflect'),
                                                        torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), 
                                                        torchvision.transforms.Normalize(mean,std)]) 
    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                        torchvision.transforms.Normalize(mean,std)]) 
    cifar100_train_dset = torchvision.datasets.CIFAR100('./cifar100_pytorch', train=True, transform=transform_train, download=False)
    cifar100_test_dset = torchvision.datasets.CIFAR100('./cifar100_pytorch', train=False, transform=transform_test, download=False)