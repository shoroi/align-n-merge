import os
import torch
import argparse

from core.utils import prepare_experiment_config, reset_bn_stats, evaluate_model, CONCEPT_TASKS, write_to_csv, SpoofModel, inject_multi_model_set, seed_everything
    

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
    
    args = parser.parse_args()

    return args


def evaluate_pair_models(eval_type, models, config, csv_file):
    train_loader = config['data']['train']['full']
    models = [reset_bn_stats(base_model, train_loader) for base_model in config['models']['bases']]
    models = config['models']['bases']

    if len(sum(config['dataset']['class_splits'], [])) != len(set(sum(config['dataset']['class_splits'], []))):
        print('Running same classes evaluation scenario, only affects logit evaluation.', flush=True)
        same_classes = True
    else:
        same_classes = False
    
    for idx, model in enumerate(models):
        results = evaluate_model(eval_type, model, config, same_classes=same_classes)
        results['Model'] = CONCEPT_TASKS[idx]
        write_to_csv(results, csv_file=csv_file)
        print(results)
    
    ensembler = SpoofModel(models)
    results = evaluate_model(eval_type, ensembler, config, same_classes=same_classes)
    results['Model'] = 'Ensemble'
    write_to_csv(results, csv_file=csv_file)
    print(results)


def get_config_from_exp_dir(exp_dir):
    if 'none' in exp_dir:
        data_model_details, split_bs_details = exp_dir.split('/')[-2:]

        if len(split_bs_details.split('_')) == 2:
            data_split, bs = split_bs_details.split('_')
        elif len(split_bs_details.split('_')) == 3:
            data_split, bs, weight_sampler = split_bs_details.split('_')
            weight_sampler = True

        exp_name = '/'.join(exp_dir.split('/')[-2:])
    else:
        data_model_details, split_bs_details, data_seed_for_split = exp_dir.split('/')[-3:]

        if len(split_bs_details.split('_')) == 3:
            data_split, bs, num_splits = split_bs_details.split('_')
        elif len(split_bs_details.split('_')) == 4:
            data_split, bs, num_splits, weight_sampler = split_bs_details.split('_')
            weight_sampler = True

        num_splits = int(num_splits[:-6])  # Extracts num splits as int
        exp_name = '/'.join(exp_dir.split('/')[-3:])

    bs = int(bs[2:])  # Extracts batch size as int
    dataset_name, model_name, eval_type = data_model_details.split('_')

    config = {}
    config['dataset'] = {}
    config['dataset']['name'] = dataset_name
    config['dataset']['shuffle_train'] = True
    config['dataset']['num_classes'] = int(dataset_name[5:])

    config['model'] = {}
    config['model']['name'] = model_name
    config['model']['dir'] = exp_dir
    config['model']['bases'] = []

    config['eval_type'] = eval_type

    config['data_split'] = data_split
    config['exp_name'] = exp_name

    return config


def modify_none_model_set(raw_config, model_set, seeds):
    if len(seeds) > 5:
        raw_config['exp_name'] = f'{raw_config["exp_name"]}/num_models_{len(seeds)}/seeds_{"-".join(seeds)}'
    else:
        raw_config['exp_name'] = f'{raw_config["exp_name"]}/seeds_{"-".join(seeds)}'

    # model_set contains all trained seeds, if we want to evaluate just some of them (those in SEED)
    # we need to remove the names that don't have one of the seeds in SEEDS
    model_bases_to_remove = []

    for model_base in model_set:
        remove_flag = True
        for seed in seeds:
            # if seed in model_base:
            if seed == model_base.split('seed')[-1]:
                # print(f'Keeping base {model_base}', flush=True)
                remove_flag = False
                break
        
        if remove_flag: 
            model_bases_to_remove.append(model_base)
    
    model_set = [model_base for model_base in model_set if model_base not in model_bases_to_remove]
    print(f'Final model set: {model_set}')
    return raw_config, model_set


if __name__ == "__main__":
    args = parse_args()
    print(f'Starting base model evaluation with models in directory: {args.exp_dir}', flush=True)
    seed_everything(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    raw_config = get_config_from_exp_dir(args.exp_dir)
    raw_config['device'] = device
    raw_config['merging_fn'] = 'match_tensors_zipit'
    raw_config['merging_metrics'] = ['covariance', 'mean']

    model_set = os.listdir(raw_config['model']['dir'])

    # For no data split we can select which seeds we want to merge / evaluate
    if raw_config['data_split'] == 'none':
        raw_config, model_set = modify_none_model_set(raw_config, model_set, seeds = args.none_seeds.split('_'))

    csv_file = os.path.join(
        '/home/mila/h/horoiste/scratch/zipit_data/csvs/multi_model_merging',
        raw_config['exp_name'],
        'base_models.csv'
    )
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)    

    with torch.no_grad():
        # raw_config = inject_pair(raw_config, model_set)
        raw_config = inject_multi_model_set(raw_config, model_set)
        config = prepare_experiment_config(raw_config)  # This should be fine

        evaluate_pair_models(
            eval_type=config['eval_type'],
            models=config['models']['bases'],
            config=config,
            csv_file=csv_file
        )
