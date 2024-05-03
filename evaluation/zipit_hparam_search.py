import os
import torch
from copy import deepcopy
import argparse
from itertools import product
import numpy as np

from core.utils import prepare_experiment_config, reset_bn_stats, evaluate_model, flatten_nested_dict, write_to_csv, seed_everything, inject_multi_model_set
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


def hyperparam_search(
    config, stop_nodes, 
    search_config, 
    device, 
    csv_file=''
    ):
    if len(sum(config['dataset']['class_splits'], [])) != len(set(sum(config['dataset']['class_splits'], []))):
        print('Running same classes evaluation scenario, only affects logit evaluation.', flush=True)
        same_classes = True
    else:
        same_classes = False

    keys, values = zip(*search_config.items())
    Merge = ModelMerge(device=device)
    for stop_node in stop_nodes:
        for bundle in product(*values):  # Iterates over all possible combinations of hparams
            instance_params = dict(zip(keys, bundle))
        
            Grapher = config['graph']
            graphs = [Grapher(deepcopy(base_model)).graphify() for base_model in base_models]
            # Construct Merger and Merge Models
            # Merge = ModelMerge(*graphs, device=device)
            Merge.init(graphs, device=device)
            Merge.transform(
                deepcopy(config['models']['new']), 
                train_loader, 
                transform_fn=config['merging_fn'], 
                metric_classes=config['metric_fns'],
                stop_at=stop_node,
                **instance_params
            )
            reset_bn_stats(Merge, train_loader)
            # Create Results Dict
            results = evaluate_model(config['eval_type'], Merge, config, same_classes=same_classes)
            results.update(flatten_nested_dict(instance_params, parent_key='params', sep=' '))
            results['stop_node'] = stop_node
            results['Merge Fn'] = config['merging_fn'].__name__
            results['Time'] = Merge.compute_transform_time
            print(results)
            write_to_csv(results, csv_file=csv_file)
            

if __name__ == "__main__":
    args = parse_args()
    print(f'Starting ZipIt! hyper-parameter search with models in directory: {args.exp_dir}', flush=True)
    seed_everything(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    raw_config = get_config_from_exp_dir(args.exp_dir)
    raw_config['device'] = device
    raw_config['merging_fn'] = 'match_tensors_zipit'
    raw_config['merging_metrics'] = ['covariance', 'mean']
    raw_config['ref_model'] = args.ref_model
    
    stop_nodes = [None]
    search_config = {
        'a':[.0001, .01, .1, .3, .5, 1.][::-1],  # [::-1] simply reverses the order
        'b': [.0, .0375, .075, .125, .25, 1.][::-1]
    }
    model_set = os.listdir(raw_config['model']['dir'])

    # For no data split we can select which seeds we want to merge / evaluate
    if raw_config['data_split'] == 'none':
        raw_config, model_set = modify_none_model_set(raw_config, model_set, seeds = args.none_seeds.split('_'))

    csv_file = os.path.join(
        '/home/mila/h/horoiste/scratch/zipit_data/csvs/multi_model_merging',
        raw_config['exp_name'],
        'zipit_hyperparameters.csv'
    )
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    raw_config = inject_multi_model_set(raw_config, model_set)
    config = prepare_experiment_config(raw_config)
    
    with torch.no_grad():
        train_loader = config['data']['train']['full']
        base_models = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]

        if len(sum(config['dataset']['class_splits'], [])) != len(set(sum(config['dataset']['class_splits'], []))):
            print('Running same classes evaluation scenario, only affects logit evaluation.', flush=True)
            same_classes = True
        else:
            same_classes = False

        base_model_accs = [evaluate_model(config["eval_type"], base_model, config, same_classes=same_classes)['Joint'] for base_model in base_models]
        print(f'Base models accuracies before reference model swap: {base_model_accs}', flush=True)
        best_model_pos = int(np.argmax(base_model_accs))
        worst_model_pos = int(np.argmin(base_model_accs))
        if raw_config['ref_model']=='best' and best_model_pos !=0:
            base_models[0], base_models[best_model_pos] = base_models[best_model_pos], base_models[0]
        elif raw_config['ref_model']=='worst' and worst_model_pos !=0:
            base_models[0], base_models[worst_model_pos] = base_models[worst_model_pos], base_models[0]
        
        import pickle
        # Serializes and then de-serializes the model, this effectively creates a copy of each initial model (modifying base_model will not affects its copy in evaled_models)
        evaled_models = [pickle.loads(pickle.dumps(base_model)) for base_model in base_models]
        for i, base_model in enumerate(evaled_models):
            # base_acc = evaluate_logits(  # Hard-coded evaluation of logits (instead of being flexible and allowing CLIP embeddings as well)
            #     base_model, config['data']['test']['splits'][i], use_flip_aug=False, 
            # )

            base_acc = evaluate_model(config['eval_type'], base_model, config, same_classes=same_classes)
            print('Base Model {} Acc: {}'.format(i, base_acc))
        
        for b, e in zip(base_models, evaled_models):
            # Prints the keys of parameters that are different between the base model and the copies, shouldn't be any
            print([k for (k, v) in b.state_dict().items() if (v != e.state_dict()[k]).sum() > 0])
        
        hyperparam_search(
            config=config, 
            stop_nodes=stop_nodes,
            search_config=search_config,
            device=device,
            csv_file=csv_file
        )
