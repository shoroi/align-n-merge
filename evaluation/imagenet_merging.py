from copy import deepcopy
import time
import argparse

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset

from core.utils import *
from core.model_merger import ModelMerge
from core.matching_functions import *
from evaluation.base_ensemble_eval import get_config_from_exp_dir, modify_none_model_set
from evaluation.weight_matching_resnets import *
import graphs.resnet_graph as graphs
from models.resnets import resnet20
from training.train_imagenet import Summary, AverageMeter, accuracy, ProgressMeter, replace_layers_agnostic, get_class_subset_indices


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--merging-fn', type=str)
parser.add_argument('--model0-dir', type=str)
parser.add_argument('--model1-dir', type=str)
parser.add_argument('--gamma',
                        type=float,
                        default=1,
                        help="Override the gamma hparam for CCA concept merging")
parser.add_argument('--hparam-a',
                        type=float,
                        default=1,
                        help="Override the a hparam for ZipIt! concept merging")
parser.add_argument('--hparam-b',
                        type=float,
                        default=1,
                        help="Override the b hparam for ZipIt! concept merging")
parser.add_argument('--num-classes-subset',
                        type=int,
                        default=0,
                        help='Number of classes to use for subset, otherwise assumes full val set.')

class Ensembler(torch.nn.Module):
    """wrap model, allow for multiple forward passes at once."""
    def __init__(self, models):
        super().__init__()
        self.models = models
        
    def forward(self, x):
        """Call all models returning list of their outputs."""
        preds = torch.stack([model(x) for model in self.models], dim=0)
        return preds.mean(dim=0)
    
    def parameters(self):
        """Return list of parameters from first model."""
        return self.models[0].parameters()

def validate(val_loader, model, criterion=torch.nn.CrossEntropyLoss()):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                # if args.gpu is not None and torch.cuda.is_available():
                #     images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    images = images.cuda(None, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(None, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    # if args.distributed:
    #     top1.all_reduce()
    #     top5.all_reduce()

    # if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
    #     aux_val_dataset = Subset(val_loader.dataset,
    #                              range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
    #     aux_val_loader = torch.utils.data.DataLoader(
    #         aux_val_dataset, batch_size=args.batch_size, shuffle=False,
    #         num_workers=args.workers, pin_memory=True)
    #     run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg

if __name__ == '__main__':
    args = parser.parse_args()

    ####################
    # Loading the data
    ####################
    traindir = '/network/datasets/imagenet.var/imagenet_torchvision/train'
    valdir = '/network/datasets/imagenet.var/imagenet_torchvision/val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # Load the indices
    if args.num_classes_subset > 0:
        train_indices = get_class_subset_indices('imagenet_train', args.num_classes_subset)
        val_indices = get_class_subset_indices('imagenet_val', args.num_classes_subset)

    if args.merging_fn == 'match_tensors_zipit':
        batch_size=32
        num_workers=4
    else:
        batch_size=256
        num_workers=8

    train_loader = torch.utils.data.DataLoader(
        Subset(train_dataset, train_indices) if args.num_classes_subset>0 else train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        Subset(val_dataset, val_indices) if args.num_classes_subset>0 else val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, sampler=None)

    ####################
    # Loading the models
    ####################
    device = torch.device('cuda:0')
    # model0_dir = '/home/mila/h/horoiste/scratch/zipit_data/checkpoints/imagenet_subset200_checkpoints/resnet18x4_bs256_lr-wu_ca_seed0'
    # model1_dir = '/home/mila/h/horoiste/scratch/zipit_data/checkpoints/imagenet_subset200_checkpoints/resnet18x4_bs256_lr-wu_ca_seed1'

    # # First Version
    # model0_dir = os.path.join('/home/mila/h/horoiste/scratch/zipit_data/checkpoints/imagenet_subset200_checkpoints/', args.model0_dir)
    # model1_dir = os.path.join('/home/mila/h/horoiste/scratch/zipit_data/checkpoints/imagenet_subset200_checkpoints/', args.model1_dir)

    # arch = model0_dir.split('/')[-1].split('_')[0]

    # model0_sd_file = os.path.join(model0_dir, 'model_best.pth.tar')
    # model1_sd_file = os.path.join(model1_dir, 'model_best.pth.tar')

    model0_sd_file = args.model0_dir
    model1_sd_file = args.model1_dir
    arch = model0_sd_file.split('/')[-2].split('_')[1]


    if len(arch.split('x')) > 1:
        w = int(arch.split('x')[1])
    else:
        w = 1

    if 'resnet20' in arch:
        model0 = resnet20(w=w, num_classes=1000)
        model1 = resnet20(w=w, num_classes=1000)
        merged_model = resnet20(w=w, num_classes=1000)
        Grapher = graphs.resnet20
        permutation_spec = resnet20_permutation_spec()
    elif 'resnet18' in arch:
        model0 = models.resnet18()
        model1 = models.resnet18()
        merged_model = models.resnet18()
        Grapher = graphs.resnet18
        permutation_spec = resnet18_permutation_spec()
        if w > 1:
            print('Using custom replace_layers_agnostic function to increase model width', flush=True)
            replace_layers_agnostic(model0, scale=w)
            replace_layers_agnostic(model1, scale=w)
            replace_layers_agnostic(merged_model, scale=w)

    # print(model0.state_dict().keys())
    # print(torch.load(model0_sd_file).keys())
    sd0 = torch.load(model0_sd_file)['state_dict']
    sd0 = {key.replace("module.", ""): value for key, value in sd0.items()}
    model0.load_state_dict(sd0)
    sd1 = torch.load(model1_sd_file)['state_dict']
    sd1 = {key.replace("module.", ""): value for key, value in sd1.items()}
    model1.load_state_dict(sd1)

    model0.to(device).eval()
    model1.to(device).eval()
    merged_model.to(device).eval()

    print(next(iter(model0.parameters())).device, flush=True)
    # Resetting BN stats (don't do this for base models takes a long time)
    # reset_bn_stats(model0, train_loader)
    # reset_bn_stats(model1, train_loader)

    print(f'Model 0 accuracy:', flush=True)
    validate(val_loader, model0)
    print(f'Model 1 accuracy:', flush=True)
    validate(val_loader, model1)


    ####################
    # Merging
    ####################
    if args.merging_fn == 'ensemble':
        ensemble = Ensembler([model0, model1])
        print(f'Ensemble accuracy:', flush=True)
        validate(val_loader, ensemble)
    elif args.merging_fn == 'weight_matching':
        rng = torch.manual_seed(123)
        start_time = time.time()
        final_permutation = weight_matching(rng, permutation_spec, get_named_params(model0.cpu()), get_named_params(model1.cpu()))
        model1_aligned = apply_permutation(permutation_spec, final_permutation,  get_named_params(model1))
        merged_model = mix_weights(merged_model, get_named_params(model0), model1_aligned, alpha=0.5).cuda()
        merged_model = reset_bn_stats(merged_model, train_loader)
        print(f'Completed {args.merging_fn} merging in: {time.time() - start_time}')

        reset_bn_stats(merged_model, train_loader)
        final_acc = validate(val_loader, merged_model)
        print(f'Accuracy with {args.merging_fn}:\n{final_acc}', flush=True)
    else:
        graphs = [Grapher(deepcopy(base_model)).graphify() for base_model in [model0, model1]]
        Merge = ModelMerge(*graphs, device=device)

        if args.merging_fn == 'match_tensors_cca':
            params = {'gamma': args.gamma}
        elif args.merging_fn == 'match_tensors_zipit':
            params = {'a': args.hparam_a, 'b': args.hparam_b}
        else:
            params = {}

        if args.merging_fn == 'match_tensors_wasserstein_acts':
            metrics = ['activs_dists']
        else:
            metrics = ['covariance', 'mean']

        start_time = time.time()
        Merge.transform(model=merged_model,
                        dataloader=train_loader,
                        transform_fn=get_merging_fn(args.merging_fn),
                        metric_classes=get_metric_fns(metrics),
                        stop_at=None,
                        save_transforms = False,
                        save_covariances= False,
                        save_dir=None,
                        **params    
        )
        print(f'Completed {args.merging_fn} merging in: {time.time() - start_time}')

        reset_bn_stats(Merge, train_loader)
        final_acc = validate(val_loader, Merge)
        print(f'Accuracy with {args.merging_fn} and params {params}:\n{final_acc}', flush=True)
        del Merge
        for i in graphs:
            del i
        del graphs
