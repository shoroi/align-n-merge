import os
import torch
import argparse
import json

import torch
import torchvision
import torchvision.transforms as T
import numpy as np

from core.utils import load_clip_features, train_cliphead, train_logits, seed_everything
from models.resnets import resnet14, resnet20
from models.vgg import vgg11, vgg16

def parse_args():
    parser = argparse.ArgumentParser(description="Train a vision model on a given dataset.")
    parser.add_argument('--base-save-dir',
                        type=str,
                        default='/home/mila/h/horoiste/scratch/zipit_data/checkpoints/multi_model_merging')
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar100',
                        choices=['cifar100', 'cifar10'],
                        help='which dataset to use.')
    parser.add_argument('--data-split',
                        type=str,
                        default='none',
                        choices=['none', 'unbalanced', 'classes', '80-20'],
                        help='''How to split the dataset.
                            none: no split, all models are trained on the entire dataset
                            unbalanced: all models are trained on all classes but with an unbalanced split of the examples
                            classes: each model is trained on a different, disjoint subset of the classes''')
    parser.add_argument('--num-splits',
                        type=int,
                        default=2,
                        help='Number of times to split the dataset')
    parser.add_argument('--data-seed',
                        type=int,
                        default=0,
                        help='Random seed that controls the dataset split.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=500,
                        help='Batch size to use.')
    parser.add_argument('--use-weighted-sampler',
                        action='store_true',
                        default=False,
                        help='Whether to use a weighted sampler to over-sample under-represented classes. Only works for unbalanced split.')
    parser.add_argument('--model',
                        type=str,
                        default='resnet20',
                        choices=['resnet14', 'resnet20', 'vgg11', 'vgg16', 'vit'],
                        help='Which model to use.')
    parser.add_argument('--model-width',
                        type=int,
                        default=8,
                        help='Width of model.')
    parser.add_argument('--use-logits',
                        action='store_true',
                        default=False,
                        help='Whether to use logits instead of CLIP features to train the models.')
    parser.add_argument('--model-seed',
                        type=int,
                        default=0,
                        help='Random seed that controls the model inits.')
    
    args = parser.parse_args()

    args_dict = vars(args)
    args_string = "\n".join([f"{key}: {value}" for key, value in args_dict.items()])
    
    return args, args_string

def get_unbalanced_split_indices(dataset, num_splits):
    # Gets the dataset and returns a list of num_splits torch.tensors containing indices for an unbalanced split of the data
    with open(f"./data/{dataset}_indices_per_class.json", "r") as file:
        class_indices = json.load(file)
    
    for i, class_i_indices in enumerate(class_indices):
        class_indices[i] = np.array(class_i_indices)[np.random.permutation(len(class_i_indices))]

    split_indices = [torch.empty(0) for i in range(num_splits)]
    class_frequencies = []

    for class_i_indices in class_indices:
        num_examples_in_class = len(class_i_indices)
        split_ratios = np.random.dirichlet(0.5*np.ones(num_splits))
        split_ratios = (num_examples_in_class*split_ratios).astype(int)
        if sum(split_ratios) < num_examples_in_class:
            # If converting to int makes it so that the split ratios don't sum to the number of examples per class
            for rest in range(num_examples_in_class - sum(split_ratios)):
                # Randomly distribute the remaining number of examples accross the splits
                split_ratios[np.random.randint(0, num_splits)] += 1
        
        class_frequencies.append(split_ratios)
        class_chunks = torch.split(torch.tensor(class_i_indices), list(split_ratios))

        for j in range(len(split_indices)):
            split_indices[j] = torch.concatenate([split_indices[j], class_chunks[j]]).type(torch.int64)
    
    class_frequencies = np.array(class_frequencies).T
    example_weights = [[] for i in range(num_splits)]
    for i in range(num_splits):
        for j in range(len(class_indices)):
            class_frequency = class_frequencies[i, j]
            class_weight = 1 / (class_frequency + 1e-5)
            example_weights[i].extend([class_weight for i in range(class_frequency)])

    return split_indices, class_frequencies, example_weights

def get_80_20_split_indices(dataset):
    with open(f"./data/{dataset}_indices_per_class.json", "r") as file:
        class_indices = json.load(file)

    for i, class_i_indices in enumerate(class_indices):
        class_indices[i] = torch.tensor(np.array(class_i_indices)[np.random.permutation(len(class_i_indices))])

    split_indices = [torch.empty(0) for i in range(2)]

    for i, class_i_indices in enumerate(class_indices):
        num_examples_in_class = len(class_i_indices)
        if i < len(class_indices)/2:
            split_indices[0] = torch.concatenate([split_indices[0], class_i_indices[:int(0.2*num_examples_in_class)]]).type(torch.int64)
            split_indices[1] = torch.concatenate([split_indices[1], class_i_indices[int(0.2*num_examples_in_class):]]).type(torch.int64)
        else:
            split_indices[0] = torch.concatenate([split_indices[0], class_i_indices[int(0.2*num_examples_in_class):]]).type(torch.int64)
            split_indices[1] = torch.concatenate([split_indices[1], class_i_indices[:int(0.2*num_examples_in_class)]]).type(torch.int64)

    return split_indices

if __name__ == "__main__":
    args, args_string = parse_args()
    print(f'Starting training with following arguments:\n{args_string}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Saving directory
    # This is just to keep the different experiments separate, model info is repeated in saved state_dict file
    # An experiment (folder) should define a specific split / training setting, where accuracies should be comparable
    exp_dir = f'{args.base_save_dir}/{args.dataset}_{args.model}x{args.model_width}_{"logits" if args.use_logits else "clip"}'
    if args.use_weighted_sampler:
        exp_dir = f'{exp_dir}_weighted-sampler'
    exp_dir = f'{exp_dir}/{args.data_split}_bs{args.batch_size}'
    if args.data_split != 'none':
        exp_dir = f'{exp_dir}_{args.num_splits}splits'

    if args.data_split != 'none':
        exp_dir = os.path.join(exp_dir, f'data_seed_{args.data_seed}')

    print(f'Save dir: {exp_dir}', flush=True)
    os.makedirs(exp_dir, exist_ok=True)

    epochs = 200                            # train epochs

    # Datasets and dataloaders
    if args.dataset == 'cifar100':
        normalize = T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])

        data_dir = './data/cifar-100-python'
        wrapper = torchvision.datasets.CIFAR100
        num_classes = 100

    elif args.dataset == 'cifar10':
        normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])

        data_dir = './data/cifar-10-python'
        wrapper = torchvision.datasets.CIFAR10
        num_classes = 10

    train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize])
    test_transform = T.Compose([T.ToTensor(), normalize])
    train_dset = wrapper(root=data_dir, train=True, download=True, transform=train_transform)
    test_dset = wrapper(root=data_dir, train=False, download=True, transform=test_transform)
        
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, num_workers=8)


    # CLIP features for dataset classes
    if args.use_logits:
        out_dim = num_classes
    else:
        clip_features = load_clip_features(test_dset.classes, device=device)
        out_dim = 512

    # Split the dataset
    seed_everything(args.data_seed)

    if args.data_split == 'none':
        # Get split dataloaders (that contain just the standard dataloaders)
        args.num_splits = 1

        train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        split_trainers = [train_loader]
        split_testers = [test_loader]

        if not args.use_logits:
            # Get label remapping
            label_remapping = np.arange(num_classes, dtype=int)

            # Get CLIP class vectors
            class_vectors = [clip_features for split in range(args.num_splits)]

    elif args.data_split == 'classes':
        # Get random split of classes
        perm_classes = np.random.permutation(num_classes)
        splits = np.array_split(perm_classes, args.num_splits)

        # Get split dataloaders
        split_trainers = [
                torch.utils.data.DataLoader(
                    torch.utils.data.Subset(
                        train_dset, [i for i, label in enumerate(train_dset.targets) if label in split]
                        ), 
                    batch_size=args.batch_size, shuffle=True, num_workers=8) for split in splits
        ]

        split_testers = [
                torch.utils.data.DataLoader(
                    torch.utils.data.Subset(
                        test_dset, 
                        [i for i, label in enumerate(test_dset.targets) if label in split]
                        ), batch_size=args.batch_size, shuffle=False, num_workers=8
                    ) for split in splits
        ]

        # Get label remapping
        label_remapping = np.zeros(num_classes, dtype=int)
        for split_idx, split in enumerate(splits):
            label_remapping[split] = np.arange(len(split))
            print(f'Split {split_idx}: {split}', flush=True)
        print("label remapping: {}".format(label_remapping), flush=True)

        # Get CLIP class vectors
        if not args.use_logits:
            class_vectors = [clip_features[split] for split in splits]
        
    elif args.data_split == 'unbalanced':
        # Get split dataloaders
        split_indices, class_frequencies, example_weights = get_unbalanced_split_indices(args.dataset, args.num_splits)

        split_trainers = [
                torch.utils.data.DataLoader(
                    torch.utils.data.Subset(train_dset, split),
                    batch_size=args.batch_size, shuffle=False if args.use_weighted_sampler else True, num_workers=8,
                    sampler=torch.utils.data.WeightedRandomSampler(weights=example_weights[split_idx], num_samples=len(example_weights[split_idx])) if args.use_weighted_sampler else None
                    ) for split_idx, split in enumerate(split_indices)
        ]

        print(f'Num examples in each split: {[len(split) for split in split_indices]}', flush=True)
        print(f'Class frequencies in each split:\n{class_frequencies}')

        split_testers = [test_loader for split in split_indices]  # Just the original test set

        if not args.use_logits:
            # Get label remapping
            label_remapping = np.arange(num_classes, dtype=int)

            # Get CLIP class vectors
            class_vectors = [clip_features for split in range(args.num_splits)]

    elif args.data_split == '80-20':
        # Get split dataloaders
        split_indices = get_80_20_split_indices(args.dataset)

        split_trainers = [
                torch.utils.data.DataLoader(
                    torch.utils.data.Subset(train_dset, split),
                    batch_size=args.batch_size, shuffle=False if args.use_weighted_sampler else True, num_workers=8
                    ) for split_idx, split in enumerate(split_indices)
        ]

        print(f'Num examples in each split: {[len(split) for split in split_indices]}', flush=True)

        split_testers = [test_loader for split in split_indices]  # Just the original test set

        if not args.use_logits:
            # Get label remapping
            label_remapping = np.arange(num_classes, dtype=int)

            # Get CLIP class vectors
            class_vectors = [clip_features for split in range(args.num_splits)]
    
    if not args.use_logits: 
        label_remapping = torch.from_numpy(label_remapping)

    for i in range(args.num_splits):
        seed_everything(args.model_seed + i)

        if args.model == 'resnet20':
            model = resnet20(w=args.model_width, num_classes=out_dim).cuda().train()
        elif args.model == 'resnet14':
            model = resnet14(w=args.model_width, num_classes=out_dim).cuda().train()
        elif args.model == 'vgg11':
            model = vgg11(w=args.model_width, num_classes=out_dim).cuda().train()
        elif args.model == 'vgg16':
            model = vgg16(w=args.model_width, num_classes=out_dim).cuda().train()
            
        if args.use_logits:
            model, final_acc = train_logits(
                model=model, train_loader=split_trainers[i], 
                test_loader=split_testers[i], epochs=epochs
            )
        else:
            model, final_acc = train_cliphead(
                model=model, train_loader=split_trainers[i], test_loader=split_testers[i], 
                class_vectors=class_vectors[i], remap_class_idxs=label_remapping, epochs=epochs
            )
                
        print(f'Base model on test split {i+1}/{args.num_splits} Acc: {final_acc}', flush=True)

        # Saving
        split_folder = f'model-seed{args.model_seed + i}'

        if args.data_split != 'none':
            split_folder = f'split{i}_{split_folder}'
            if args.data_split == 'classes':
                idxs = [str(k) for k in splits[i]]
                split_folder = f'{split_folder}_{"_".join(idxs)}'
        
        save_dir = os.path.join(exp_dir, split_folder)
        os.makedirs(save_dir, exist_ok=True)

        # save_path = os.path.join(save_dir, f'model_seed{args.model_seed + i}/state_dict_v{len(os.listdir(save_dir))}.pth.tar')
        save_path = os.path.join(save_dir, f'state_dict_v{len(os.listdir(save_dir))}.pth.tar')
        torch.save(model.state_dict(), save_path)
                
    print('Done!')
