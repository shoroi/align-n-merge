from collections import defaultdict
from typing import NamedTuple

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def get_named_params(model):
    return {name: param for name, param in model.named_parameters()}

class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict

def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    '''Creates the perm_to_axes from axes_to_perm and returns them both in a PermutationSpec object'''
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)

def conv(name, p_in, p_out):
    # return {f"{name}/kernel": (None, None, p_in, p_out)}
    return {f"{name}.weight": (p_out, p_in, None, None)}

def norm(name, p):
    # return {f"{name}/scale": (p, ), f"{name}/bias": (p, )}  # Original version
    return {f"{name}.weight": (p, ), f"{name}.bias": (p, )}  # First version (without permuting running mean and var)
    # return {f"{name}.weight": (p, ), f"{name}.bias": (p, ), f"{name}.running_mean": (p, ), f"{name}.running_var": (p, )}

def dense(name, p_in, p_out):
    # return {f"{name}/kernel": (p_in, p_out), f"{name}/bias": (p_out, )}
    return {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}

def easyblock(name, p):
    return {
        **conv(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm(f"{name}.bn1", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p),
        **norm(f"{name}.bn2", p)
    }

def shortcutblock(name, p_in, p_out, shortcut_name = 'shortcut'):
    return {
        **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}.bn1", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
        **norm(f"{name}.bn2", p_out),
        **conv(f"{name}.{shortcut_name}.0", p_in, p_out),
        **norm(f"{name}.{shortcut_name}.1", p_out),
    }

def resnet20_permutation_spec() -> PermutationSpec:
    '''First creates the axes_to_perm and then uses permutation_spec_from_axes_to_perm to compute the perm_to_axes and return the final PermutationSpec object.'''

    return permutation_spec_from_axes_to_perm({
        **conv("conv1", None, "P_bg0"),
        **norm("bn1", "P_bg0"),
        #
        **easyblock("layer1.0", "P_bg0"),
        **easyblock("layer1.1", "P_bg0"),
        **easyblock("layer1.2", "P_bg0"),
        #
        **shortcutblock("layer2.0", "P_bg0", "P_bg1"),
        **easyblock("layer2.1", "P_bg1"),
        **easyblock("layer2.2", "P_bg1"),
        #
        **shortcutblock("layer3.0", "P_bg1", "P_bg2"),
        **easyblock("layer3.1", "P_bg2"),
        **easyblock("layer3.2", "P_bg2"),
        #
        **dense("linear", "P_bg2", None),
    })

def resnet18_permutation_spec() -> PermutationSpec:
    '''First creates the axes_to_perm and then uses permutation_spec_from_axes_to_perm to compute the perm_to_axes and return the final PermutationSpec object.'''

    return permutation_spec_from_axes_to_perm({
        **conv("conv1", None, "P_bg0"),
        **norm("bn1", "P_bg0"),
        #
        **easyblock("layer1.0", "P_bg0"),
        **easyblock("layer1.1", "P_bg0"),
        #
        **shortcutblock("layer2.0", "P_bg0", "P_bg1", shortcut_name='downsample'),
        **easyblock("layer2.1", "P_bg1"),
        #
        **shortcutblock("layer3.0", "P_bg1", "P_bg2", shortcut_name='downsample'),
        **easyblock("layer3.1", "P_bg2"),
        #
        **shortcutblock("layer4.0", "P_bg2", "P_bg3", shortcut_name='downsample'),
        **easyblock("layer4.1", "P_bg3"),
        #
        **dense("fc", "P_bg3", None),
    })

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k].clone()
    for axis, p in enumerate(ps.axes_to_perm[k]):
        if axis == except_axis:
            continue
        if p is not None:
            w = torch.index_select(w, axis, perm[p])
    return w

def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply a `perm` to `params`."""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}

def weight_matching(rng,
                    ps: PermutationSpec,
                    params_a,
                    params_b,
                    max_iter=100,
                    init_perm=None,
                    silent=False):
    """Find a permutation of `params_b` to make them match `params_a`."""
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}  
    # This dictionary holds, for every permutation p, the shape of the model's parameters (params_a[axes[0][0]]) at the axis where the permutation is applied (axes[0][1])
    # It only looks at the first item of the list of axes to determine the size (since it should be the same for all)
    perm = {p: torch.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
    # This is the dictionary with the permutations in a vector format (neuron i gets moved to the value at position i in the vector)
    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False
        for p_ix in torch.randperm(len(perm_names)):
            # Randomly chose a permutation
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = torch.zeros((n, n), dtype=params_a["conv1.weight"].dtype)

            for wk, axis in ps.perm_to_axes[p]:
                # Iterates through all the axes and the parameters
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                # w_a = w_a.permute(axis, 0).reshape((n, -1))
                # w_b = w_b.permute(axis, 0).reshape((n, -1))
                w_a = torch.movedim(w_a, axis, 0).reshape((n, -1))
                w_b = torch.movedim(w_b, axis, 0).reshape((n, -1))
                A += w_a @ w_b.t()

            ri, ci = linear_sum_assignment(A.cpu().detach().numpy(), maximize=True)
            assert (ri == np.arange(len(ri))).all()

            oldL = (A * torch.eye(n)[perm[p]]).sum()
            newL = (A * torch.eye(n)[ci]).sum()
            if not silent:
                print(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = torch.tensor(ci)

        if not progress:
            break

    return perm

def mix_weights(net, sd0, sd1, alpha=0.5):
    # sd0 = model0.state_dict()
    # sd1 = model1.state_dict()
    sd_alpha = {k: (1 - alpha) * sd0[k].cuda() + alpha * sd1[k].cuda()
                for k in sd0.keys()}
    net.load_state_dict(sd_alpha, strict=False)

    return net
