import torch
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import json

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

################################################# Global Variables #################################################

normalize_cifar100 = T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
normalize_cifar10 = T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])

####################################################################################################################

def get_subset_indices(config, ratio=0.1):
    # Returns the indices for a balanced (equal num examples per class) subset of the dataset of size ratio*DATASET_SIZE.
    num_classes = config['num_classes']
    num_examples_per_class = int(50000/num_classes)
    num_examples_per_class_to_keep = int(num_examples_per_class * ratio + 0.1)

    with open(f"./data/{config['type']}{config['num_classes']}_indices_per_class.json", "r") as file:
        class_indices = json.load(file)
    
    subset_indices = []
    for i in range(num_classes):
        class_indices[i] = torch.tensor(class_indices[i])[np.random.permutation(num_examples_per_class)]
        subset_indices.append(class_indices[i][:num_examples_per_class_to_keep])
    
    return torch.concatenate(subset_indices)


def get_correct_normalize(config):
    if config['num_classes'] == 100:
        return normalize_cifar100
    else:
        return normalize_cifar10


def prepare_train_loaders(config):
    normalize = get_correct_normalize(config)
    if 'no_transform' in config:
        train_transform = T.Compose([T.ToTensor(), normalize])
    else:
        train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize])

    train_dset = config['wrapper'](root=config['dir'], train=True, download=True, transform=train_transform)
    loaders = {'full': torch.utils.data.DataLoader(train_dset, batch_size=config['batch_size'], shuffle=config['shuffle_train'], num_workers=config['num_workers'])}
    
    if 'class_splits' in config:
        loaders['splits'] = []
        grouped_class_indices = np.zeros(config['num_classes'], dtype=int)
        for i, splits in enumerate(config['class_splits']):
            valid_examples = [i for i, label in tqdm(enumerate(train_dset.targets)) if label in splits]  # Extracts indices if label is in class split
            data_subset = torch.utils.data.Subset(train_dset, valid_examples)
            loaders['splits'].append(torch.utils.data.DataLoader(
                data_subset, batch_size=config['batch_size'], shuffle=config['shuffle_train'], num_workers=config['num_workers']
            ))
            grouped_class_indices[splits] = np.arange(len(splits))

        loaders["label_remapping"] = torch.from_numpy(grouped_class_indices)
        loaders['class_splits'] = config['class_splits']
    
    subset_indices = get_subset_indices(config)
    loaders['subset'] = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dset, subset_indices),
        batch_size = len(subset_indices),
        shuffle=config['shuffle_train'], 
        num_workers=config['num_workers']
    )

    return loaders


def prepare_test_loaders(config):
    normalize = get_correct_normalize(config)
    test_transform = T.Compose([T.ToTensor(), normalize])
    test_dset = config['wrapper'](root=config['dir'], train=False, download=True, transform=test_transform)
    loaders = {'full': torch.utils.data.DataLoader(test_dset, batch_size=config['batch_size'], shuffle=config["shuffle_test"], num_workers=config['num_workers'])}
    
    if 'class_splits' in config:
        loaders['splits'] = []
        for i, splits in enumerate(config['class_splits']):
            valid_examples = [i for i, label in tqdm(enumerate(test_dset.targets)) if label in splits]
            data_subset = torch.utils.data.Subset(test_dset, valid_examples)
            loaders['splits'].append(torch.utils.data.DataLoader(
                data_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers']
            ))
            
    loaders['class_names'] = test_dset.classes
    return loaders
