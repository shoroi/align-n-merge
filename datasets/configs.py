import torchvision

cifar100 = {
    'dir': './data/cifar-100-python', 
    'num_classes': 100,
    'wrapper': torchvision.datasets.CIFAR100,
    'batch_size': 500,
    'type': 'cifar',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 2,
}

cifar10 = {
    'dir': './data/cifar-10-python',
    'num_classes': 10,
    'wrapper': torchvision.datasets.CIFAR10,
    'batch_size': 500,
    'type': 'cifar',
    'shuffle_train': True,
    'shuffle_test': False,
    'num_workers': 8,
}
