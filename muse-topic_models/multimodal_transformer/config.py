import os, csv
import torch

from src.utils import *

####################################################################
#
# Parameters/ Hyperparameters
#
####################################################################

def load_config_base(args):
    params = {
        'model': 'MulT',
        'experiment_name':args.experiment_name,
        'vonly': False,
        'aonly': False,
        'lonly': False,
        'aligned': args.aligned,
        'dataset': 'muse',
        'data_path': 'data',
        'attn_dropout': 0.1,
        'attn_dropout_a': 0.0,
        'attn_dropout_v': 0.0,
        'relu_dropout': 0.1,
        'embed_dropout': 0.25,
        'res_dropout': 0.1,
        'out_dropout': 0.0,
        'nlevels': 5,
        'num_heads': 5,
        'attn_mask': True,
        'batch_size': 16,
        'clip': 0.8,
        'lr': 1e-3,
        'optim': 'Adam',
        'num_epochs': 20,
        'when': 20,
        'batch_chunk': 1,
        'log_interval': 30,
        'seed': 1,
        'no_cuda': False,
        'name': 'mult',
    }



    return params

def dim_criterion(class_name, regression_mode):
    if class_name in ['arousal','valence']:
        if regression_mode:
            criterion_dict = {
                'muse':'L1Loss'
            } 
            output_dim_dict = {
                'muse':1
            }
        else:
            criterion_dict = {
                'muse':'CrossEntropyLoss'
            } 
            output_dim_dict = {
                'muse':3
            }
    elif class_name == 'topic':
        criterion_dict = {
            'muse':'CrossEntropyLoss'
        } 
        output_dim_dict = {
            'muse':10
        }
    return output_dim_dict, criterion_dict


def check_cuda(params):
    use_cuda = False
    torch.set_default_tensor_type('torch.FloatTensor')
    if torch.cuda.is_available():
        if params['no_cuda']:
            print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
        else:
            torch.cuda.manual_seed(params['seed'])
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            use_cuda = True
    return use_cuda


def load_config(args):
    params = load_config_base(args)

    params['class_name'] = args.class_name
    params['cont_emotions'] = args.cont_emotions
    params['regression_mode'] = args.regression_mode

    torch.manual_seed(params['seed'])
    dataset = str.lower(params['dataset'].strip())
    valid_partial_mode = params['lonly'] + params['vonly'] + params['aonly']

    if valid_partial_mode == 0:
        params['lonly'] = params['vonly'] = params['aonly'] = True
    elif valid_partial_mode != 1:
        raise ValueError("You can only choose one of {l/v/a}only.")

    params['layers'] = params['nlevels']
    params['use_cuda'] = check_cuda(params)
    params['dataset'] = dataset
    params['when'] = params['when']
    params['batch_chunk'] = params['batch_chunk']
    params['model'] = str.upper(params['model'].strip())

    # added
    params['y_names'] = get_class_names(args.class_name)
    params['class_name'] = args.class_name
    params['output_dim'] = dim_criterion(args.class_name, args.regression_mode)[0].get(dataset, 1)
    params['criterion'] = dim_criterion(args.class_name, args.regression_mode)[1].get(dataset, 'L1Loss')
    source_filename = "_".join([args.vision,args.audio,args.text])
    if args.aligned:
        source_filename += '_data'
    else:
        source_filename += '_data_noalign'
    if args.cont_emotions:
        source_filename += '_cont'
    params['source_name'] = "_".join(source_filename.split('_'))
    params['experiment_dir'] = os.path.join('experiments')
    params['output_dir'] = os.path.join(params['experiment_dir'],'results',params['class_name'], args.experiment_name + '_' + params['source_name'])
    params['pretrained_model_dir'] = os.path.join(params['experiment_dir'],'pretrained_model',params['class_name'], args.experiment_name + '_' + params['source_name'])
    if not os.path.exists(params['pretrained_model_dir']):
        os.makedirs(params['pretrained_model_dir'])
    if not os.path.exists(params['output_dir']):
        os.makedirs(params['output_dir'])

    return params

def update_config(args, params, train_data, valid_data, test_data):

    params['orig_d_l'], params['orig_d_a'], params['orig_d_v'] = train_data.get_dim()
    params['l_len'], params['a_len'], params['v_len'] = train_data.get_seq_len()
    if args.evaluate_test:
        params['n_train'], params['n_valid'], params['n_test'] = len(train_data), len(valid_data), len(test_data)
    else:
        params['n_train'], params['n_valid'], params['n_test'] = len(train_data), len(valid_data), 0
    return params


def save_config(params):
    
    # save config
    with open(os.path.join(params['output_dir'], 'config.csv'), 'w+', newline = "") as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in params.items():
            writer.writerow([key, value])

