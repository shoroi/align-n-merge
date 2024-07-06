import os
import torch
import argparse
import pandas as pd
import numpy as np
from time import time

from copy import deepcopy

from core.utils import *
from core.model_merger import ModelMerge
from evaluation.base_ensemble_eval import get_config_from_exp_dir, modify_none_model_set


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base models on the different tasks.")
    parser.add_argument('--exp-dir',
                        type=str,
                        default=None,
                        help="Overrides config['model']['dir'] with this value to make it easier to run from command line.")
    parser.add_argument('--none-seeds',
                        type=str,
                        default='0_1_2_3_4',
                        help="Which seeds to evaluate if the data split is none.")
    parser.add_argument('--merging-fn',
                        type=str,
                        default=None,
                        choices=['match_tensors_identity', 'match_tensors_permute', 'match_tensors_cca', 'match_tensors_zipit', 'match_tensors_wasserstein_acts'])
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
    parser.add_argument('--save-transforms',
                        action='store_true',
                        default=False,
                        help='Whether to save the merge / unmerge matrices.')
    parser.add_argument('--save-covariances',
                        action='store_true',
                        default=False,
                        help='Whether to save the covariance metrics.')
    parser.add_argument('--manual-hparams',
                        action='store_true',
                        default=False,
                        help='If True will use the args hparam value otherwise it retrieves the optimal one.')
    parser.add_argument('--ref-model',
                        type=str,
                        default='random',
                        choices=['random', 'best', 'worst'])
    
    args = parser.parse_args()

    args_dict = vars(args)
    args_string = "\n".join([f"{key}: {value}" for key, value in args_dict.items()])

    return args, args_string


def run_experiment(config):
    if len(sum(config['dataset']['class_splits'], [])) != len(set(sum(config['dataset']['class_splits'], []))):
        print('Running same classes evaluation scenario, affects both CLIP and logits evaluation.', flush=True)
        same_classes = True
    else:
        same_classes = False

    train_loader = config['data']['train']['full']
    base_models = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]

    base_model_accs = [evaluate_model(config["eval_type"], base_model, config, same_classes=same_classes)['Joint'] for base_model in base_models]
    print(f'Base models accuracies before reference model swap: {base_model_accs}', flush=True)
    best_model_pos = int(np.argmax(base_model_accs))
    worst_model_pos = int(np.argmin(base_model_accs))
    if config['ref_model']=='best' and best_model_pos !=0:
        base_models[0], base_models[best_model_pos] = base_models[best_model_pos], base_models[0]
    elif config['ref_model']=='worst' and worst_model_pos !=0:
        base_models[0], base_models[worst_model_pos] = base_models[worst_model_pos], base_models[0]
        
    Grapher = config['graph']
    graphs = [Grapher(deepcopy(base_model)).graphify() for base_model in base_models]
    Merge = ModelMerge(*graphs, device=device)
    start_time = time()
    Merge.transform(
        deepcopy(config['models']['new']), 
        train_loader, 
        transform_fn=config['merging_fn'], 
        metric_classes=config['metric_fns'],
        stop_at=config['stop_at'],
        save_transforms=config['save_transforms'],
        save_covariances=config['save_covariances'],
        save_dir=config["save_dir"],
        **config['params']
    )
    print(f'Merging completed in {time() - start_time}', flush=True)
    reset_bn_stats(Merge, train_loader)
        
    results = evaluate_model(config['eval_type'], Merge, config, same_classes=same_classes)
    results['Time'] = Merge.compute_transform_time
    results['Merging Fn'] = config['merging_fn'].__name__
    results['Stop Node'] = config['stop_at']
    for key in config['params']:
        results[key] = config['params'][key]
    for idx, split in enumerate(model_set):
        results[f'Split {CONCEPT_TASKS[idx]}'] = split
    # write_to_csv(results, csv_file=os.path.join(config['save_dir'], f'{config["merging_fn"].__name__}.csv'))
    print(results)
        
    return results


def get_optimal_hparam_value(config):
    if config['data_split'] == 'none':
        seeds_str = config['save_dir'].split('_')[-1]  # Last split of '_' is the seeds in format 0-1-2
        # num_seeds = math.ceil(len_seeds_str/2)  # THIS DOESN'T WORK WHEN SEEDS HAVE 2 DIGITS
        num_seeds = len(seeds_str.split('-'))
        hparam_dir = f'{config["save_dir"][:-len(seeds_str)]}{"-".join([str(i) for i in range(num_seeds)])}'
    else:
        hparam_dir = f'{config["save_dir"][:-1]}0'  # The last element of save_dir is the data_seed, replace it with 0 to get hparam directory

    if config['merging_fn'] == 'match_tensors_cca':
        df = pd.read_csv(f'{hparam_dir}/cca_hparams.csv')
        max_acc_idx = df['Joint'].idxmax()
        return df.at[max_acc_idx, 'Gamma Value']
    elif config['merging_fn'] == 'match_tensors_zipit':
        df = pd.read_csv(f'{hparam_dir}/zipit_hyperparameters.csv')
        max_acc_idx = df['Joint'].idxmax()
        return df.at[max_acc_idx, 'Params A'], df.at[max_acc_idx, 'Params B']


if __name__ == "__main__":
    args, args_string = parse_args()
    print(f'Starting {args.merging_fn} concept merging evaluation with with arguments:\n{args_string}', flush=True)
    seed_everything(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    raw_config = get_config_from_exp_dir(args.exp_dir)
    raw_config['device'] = device
    raw_config['merging_fn'] = args.merging_fn
    if raw_config['merging_fn'] == 'match_tensors_wasserstein_acts':
        raw_config['merging_metrics'] = ['activs_dists']
    else:
        raw_config['merging_metrics'] = ['covariance', 'mean']
    raw_config['stop_at'] = None
    raw_config['save_transforms'] = args.save_transforms
    raw_config['save_covariances'] = args.save_covariances
    raw_config['ref_model'] = args.ref_model

    model_set = os.listdir(raw_config['model']['dir'])

    # For no data split we can select which seeds we want to merge / evaluate
    if raw_config['data_split'] == 'none':
        raw_config, model_set = modify_none_model_set(raw_config, model_set, seeds = args.none_seeds.split('_'))

    raw_config['save_dir'] = os.path.join('/home/mila/h/horoiste/scratch/zipit_data/csvs/multi_model_merging', raw_config['exp_name'])
    os.makedirs(raw_config['save_dir'], exist_ok=True)

    if raw_config['merging_fn'] == 'match_tensors_cca':
        if args.manual_hparams:
            raw_config['params'] = {'gamma': args.gamma}
        else:
            raw_config['params'] = {'gamma': get_optimal_hparam_value(raw_config)}
    elif raw_config['merging_fn'] == 'match_tensors_zipit':
        if args.manual_hparams:
            raw_config['params'] = {'a': args.hparam_a, 'b': args.hparam_b}
        else:
            a, b = get_optimal_hparam_value(raw_config)
            raw_config['params'] = {'a': a, 'b': b}
    else:
        raw_config['params'] = {}
    
    raw_config = inject_multi_model_set(raw_config, model_set)
    config = prepare_experiment_config(raw_config)    
    
    with torch.no_grad():
        node_results = run_experiment(config)   
