import os
import torch
import argparse
import numpy as np

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
    parser.add_argument('--ref-model',
                        type=str,
                        default='random',
                        choices=['random', 'best', 'worst'])
    
    args = parser.parse_args()

    return args


def run_auxiliary_experiment(merging_fn, experiment_config, model_set, device, stop_at=None, csv_file='', gamma=0):
    experiment_config = inject_multi_model_set(experiment_config, model_set)
    config = prepare_experiment_config(experiment_config)
    train_loader = config['data']['train']['full']
    base_models = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]

    if len(sum(config['dataset']['class_splits'], [])) != len(set(sum(config['dataset']['class_splits'], []))):
        print('Running same classes evaluation scenario, affects both CLIP and logits evaluation.', flush=True)
        same_classes = True
    else:
        same_classes = False

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
    Merge.transform(
        deepcopy(config['models']['new']), 
        train_loader, 
        transform_fn=get_merging_fn(merging_fn), 
        metric_classes=config['metric_fns'],
        stop_at=stop_at,
        **{'gamma': gamma}
    )
    reset_bn_stats(Merge, train_loader)
        
    results = evaluate_model(experiment_config['eval_type'], Merge, config, same_classes=same_classes)
    results['Time'] = Merge.compute_transform_time
    results['Merging Fn'] = merging_fn
    results['Gamma value'] = gamma
    for idx, split in enumerate(model_set):
        results[f'Split {CONCEPT_TASKS[idx]}'] = split
    write_to_csv(results, csv_file=csv_file)
    print(results)
        
    return results


if __name__ == "__main__":
    args = parse_args()
    print(f'Starting CCA concept merging hyperparameter search with models in directory: {args.exp_dir}', flush=True)
    seed_everything(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    raw_config = get_config_from_exp_dir(args.exp_dir)
    raw_config['device'] = device
    raw_config['merging_fn'] = 'match_tensors_cca'
    raw_config['merging_metrics'] = ['covariance', 'mean']
    raw_config['ref_model'] = args.ref_model

    stop_at = None
    model_set = os.listdir(raw_config['model']['dir'])

    # For no data split we can select which seeds we want to merge / evaluate
    if raw_config['data_split'] == 'none':
        raw_config, model_set = modify_none_model_set(raw_config, model_set, seeds = args.none_seeds.split('_'))

    csv_file = os.path.join(
        '/home/mila/h/horoiste/scratch/zipit_data/csvs/multi_model_merging',
        raw_config['exp_name'],
        'cca_hparams.csv'
    )
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    gamma_values = [1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    for gamma_value in gamma_values:    
        with torch.no_grad():
            node_results = run_auxiliary_experiment(
                merging_fn=raw_config['merging_fn'], 
                experiment_config=raw_config, 
                model_set=model_set, 
                device=device, 
                csv_file=csv_file,
                stop_at=stop_at,
                gamma=gamma_value
            )
