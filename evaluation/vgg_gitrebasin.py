import torch
from models.vgg import vgg11
import numpy as np
import scipy.optimize
from utils import evaluate_logits_alltasks_same_classes
import torchvision
import torchvision.transforms as T
from torch import nn


def solve_lap(corr_mtx):
    # perm_map: is a with element at position i saying where index i is going
    if not isinstance(corr_mtx, np.ndarray):
        corr_mtx_a = corr_mtx.cpu().numpy()
    else:
        corr_mtx_a = corr_mtx

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx_a, maximize=True)
    assert (row_ind == np.arange(len(corr_mtx_a))).all()
    perm_map = torch.tensor(col_ind).long()
    return perm_map

def permute_output(perm_map, layer):
    # modifies the weight matrices of a convolution and batchnorm
    # layer given a permutation of the output channels
    pre_weights = [layer.weight,
                   layer.bias]
    for w in pre_weights:
        w.data = w[perm_map]

def permute_input(perm_map, layer):
    # modifies the weight matrix of a layer for a given permutation of the input channels
    # works for both conv2d and linear 
    w = layer.weight
    w.data = w[:, perm_map]

def permute_model1_weights(model0, model1, verbose = False, return_perm_maps=False):
    feats0 = model0.features
    feats1 = model1.features
    n = len(feats0)

    permutation_maps = []
    features_to_permute = []

    for i in range(n):
        if not isinstance(feats1[i], nn.Conv2d):
            continue  # ends current iteration of the loop and goes to the next one
        assert isinstance(feats1[i+1], nn.ReLU)

        if len(features_to_permute) == 0:
            permutation_maps.append(np.arange(feats1[i].weight.shape[1]))  # add an identity permutation for the first layer
        permutation_maps.append(np.arange(feats1[i].weight.shape[0]))
        features_to_permute.append(i)
    
    convergence_criterion = 1

    while convergence_criterion > 0:
        convergence_criterion = 0
        if verbose:
            print('Starting new iteration', flush=True)
        for idx in np.random.permutation(len(features_to_permute)): # np.arange(len(features_to_permute)):
            # print(f'idx {idx}')
            layer = features_to_permute[idx]

            weights0 = feats0[layer].weight  # C_out is first dim
            weights0 = weights0.reshape((weights0.shape[0], -1))  # change shape to have C_out x C_in H W

            perm = permutation_maps[idx]
            weights1 = feats1[layer].weight
            weights1 = weights1[:, perm]
            weights1 = weights1.reshape((weights1.shape[0], -1))
            obj0 = torch.matmul(weights0, weights1.transpose(0,1))

            if idx != (len(features_to_permute)-1):
                next_layer = features_to_permute[idx+1]
                weights0 = feats0[next_layer].weight.permute((1,0,2,3))  # permute to have C_in first
                weights0 = weights0.reshape((weights0.shape[0], -1))

                perm = permutation_maps[idx + 2]
                weights1 = feats1[next_layer].weight
                weights1 = weights1[perm].permute((1,0,2,3)) # permute to have C_in first
                weights1 = weights1.reshape((weights1.shape[0], -1))
                obj1 = torch.matmul(weights0, weights1.transpose(0,1))

            else:
                weights0 = model0.classifier.weight.permute((1, 0))  # permute to have C_in first
                weights1 = model1.classifier.weight
                obj1 = torch.matmul(weights0, weights1)
            
            new_perm_map = solve_lap((obj0 + obj1).detach()).numpy()
            convergence_criterion += np.sum(np.abs(permutation_maps[idx+1] - new_perm_map))
            permutation_maps[idx+1] = new_perm_map
        if verbose:
            print(convergence_criterion)


    for idx, i in enumerate(features_to_permute):
        perm_map = permutation_maps[idx+1]
        permute_output(perm_map, feats1[i])
    
        # look for the next conv layer, whose inputs should be permuted the same way
        next_layer = None
        for j in range(i+1, n):
            if isinstance(feats1[j], nn.Conv2d):
                next_layer = feats1[j]
                break
        if next_layer is None:
            next_layer = model1.classifier
        permute_input(perm_map, next_layer)
    
    if return_perm_maps:
        return model1, permutation_maps
    else:
        return model1

def mix_weights(net, model0, model1, alpha):
    sd0 = model0.state_dict()
    sd1 = model1.state_dict()
    sd_alpha = {k: (1 - alpha) * sd0[k].cuda() + alpha * sd1[k].cuda()
                for k in sd0.keys()}
    net.load_state_dict(sd_alpha)

    return net

def mix_multiple_weights(net, models, weights=None):
    state_dicts = [model.state_dict() for model in models]
    if not weights:
        weights = np.ones(len(models)) / len(models)

    sd_alpha = {k: sum(sd[k] * w for sd, w in zip(state_dicts, weights))
                for k in state_dicts[0].keys()}
    
    net.load_state_dict(sd_alpha)
    return net


def many_merge(models, width=1):
    num_models = len(models)
    permutation_maps = [get_init_permutation_maps(model) for model in models]

    convergence_criterion = 1

    while convergence_criterion > 0:
        convergence_criterion = 0
        perm = np.random.permutation(num_models)

        for idx in perm:
            average_net = vgg11(w=width)
            average_net = mix_multiple_weights(average_net, models[:idx]+models[idx+1:])

            models[idx], new_perm_maps = permute_model1_weights(average_net, models[idx], return_perm_maps=True)

            for i in range(len(new_perm_maps)):
                convergence_criterion += np.sum(np.abs(permutation_maps[idx][i] - new_perm_maps[i]))
            permutation_maps[idx] = new_perm_maps

    return mix_multiple_weights(models)

def get_init_permutation_maps(model):
    feats = model.features
    n = len(feats)

    permutation_maps = []
    features_to_permute = []

    for i in range(n):
        if not isinstance(feats[i], nn.Conv2d):
            continue  # ends current iteration of the loop and goes to the next one
        assert isinstance(feats[i+1], nn.ReLU)

        if len(features_to_permute) == 0:
            permutation_maps.append(np.arange(feats[i].weight.shape[1]))  # add an identity permutation for the first layer
        permutation_maps.append(np.arange(feats[i].weight.shape[0]))
        features_to_permute.append(i)
    return permutation_maps

def load_model(width = 1, seed = 0):
    file = f'/home/mila/h/horoiste/scratch/zipit_data/checkpoints/multi_model_merging/cifar10_vgg11x{width}_logits/none_bs500/model-seed{seed}/state_dict_v0.pth.tar'
    model = vgg11(w=width)
    model.load_state_dict(torch.load(file))
    return model

def get_pair_interp_acc(width=1, seeds=[0,1]):
    model0 = load_model(width=width, seed=seeds[0]).cuda()
    model1 = load_model(width=width, seed=seeds[1]).cuda()

    # These seem alright, no need to re-run them
    # print(f'Model 0 acc, loss: {evaluate_logits_alltasks_same_classes(model0, test_dl, return_loss = True)}')
    # print(f'Model 1 acc, loss: {evaluate_logits_alltasks_same_classes(model1, test_dl, return_loss = True)}')

    model1 = permute_model1_weights(model0, model1)

    model_alpha = vgg11(w=width).cuda()
    # model_alpha = mix_weights(model_alpha, model0, model1, 0.5)
    model_alpha = mix_multiple_weights(model_alpha, [model0, model1])

    print(f'Merged model acc, loss: {evaluate_logits_alltasks_same_classes(model_alpha, test_dl, return_loss = True)}')

def get_multiple_interp_acc(width=1, seeds=[0,1]):
    models = [load_model(width=width, seed=seed) for seed in seeds]

    models_to_merge = [permute_model1_weights(models[0], models[i]) for i in range(1, len(seeds))]
    models_to_merge.insert(0, models[0])

    model_alpha = vgg11(w=width).cuda()
    model_alpha = mix_multiple_weights(model_alpha, models_to_merge)

    print(f'Merged model acc, loss: {evaluate_logits_alltasks_same_classes(model_alpha, test_dl, return_loss = True)}', flush=True)

def get_merge_many_interp_acc(width=1, seeds=[0,1]):
    models = [load_model(width=width, seed=seed) for seed in seeds]

    model_alpha = vgg11(w=width).cuda()
    model_alpha = many_merge(models, width=width)

    print(f'Merged model acc, loss: {evaluate_logits_alltasks_same_classes(model_alpha, test_dl, return_loss = True)}')

# GET DATALOADER
normalize_cifar10 = T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])

test_transform = T.Compose([T.ToTensor(), normalize_cifar10])
test_dset = torchvision.datasets.CIFAR10(root='/home/mila/h/horoiste/_zipit/data/cifar-10-python', train=False, download=True, transform=test_transform)
test_dl = torch.utils.data.DataLoader(test_dset, batch_size=50, shuffle=False, num_workers=8)

seeds_ = [[0, 1, 3], [0, 2, 4], [1, 4, 5], [3, 4, 5], [0, 1, 2, 4], [0, 1, 2, 5], [0, 2, 3, 4], [0, 2, 3, 5], [0, 1, 2, 3, 5], [0, 1, 3, 4, 5], [0, 1, 2, 4, 5], [1, 2, 3, 4, 5]]

for width in [8]:
    print(f'Width {width}', flush=True)
    for seeds in seeds_:
        print(f'Seeds {seeds}', flush=True)
        get_multiple_interp_acc(width=width, seeds=seeds)
