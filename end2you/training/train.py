import end2you as e2u
import models
import data_provider
import argparse

from functools import partial
from pathlib import Path


parser = argparse.ArgumentParser(description='End2You flags.')
parser.add_argument('--train_dir', type=str, default='ckpt/',
                    help='Directory where to save logs/models (default ./ckpt/train).')
parser.add_argument('--dataset_dir', type=str,
                    help='Directory with the tfrecords data.')
parser.add_argument('--learning_rate', type=float, default=.00008,
                    help='Initiale learning rate to use (default .00008).')
parser.add_argument('--batch_size', type=int, default=10,
                    help='Batch size to use during training (default 10).')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='Number of epochs to train model (default 50).')
parser.add_argument('--task', type=str,
                    help='''Which task to train for. Takes values from [1-3]: 
                              1 - MuSe-Wild, \n
                              2 - MuSe-Topic, \n
                              3 - MuSe-Trust''')
parser.add_argument('--output', type=str, default='none',
                    help='''This flag is used only for the 3rd (MuSe-Trust) sub-challenge.
                            Output should be one of ['none', 'multitask], where 'none'
                            indicates that only 'trustworthiness' will be predicted by the model, and
                            'multitask' indicates that the model will predict 'arousal', 'valence'
                            and 'trustworthiness' (default 'none').''')


def get_train_params(args):
    if args.task == '1':
        out_names = ['arousal', 'valence']
    elif args.task == '2':
        out_names = ['arousal', 'valence']
    elif args.task == '3':
        if args.output == 'none':
            out_names = ['trustworthiness']
        else:
            out_names = ['trustworthiness', 'arousal', 'valence']
    else:
        raise ValueError('')
    
    train_params = {}
    train_params['train_dir'] = args.train_dir
    train_params['initial_learning_rate'] = args.learning_rate
    train_params['num_epochs'] = args.num_epochs
    train_params['loss'] = 'sce' if args.task == '2' else 'ccc'
    train_params['pretrained_model_checkpoint_path'] = None
    train_params['input_type'] = 'audio'
    
    train_params['output_names'] = out_names
    train_params['tfrecords_eval_folder'] = Path(args.dataset_dir) / 'devel'
    train_params['save_top_k'] = 1
    
    if args.task == '3':
        train_params['data_provider'] = data_provider.get_provider(
            '3', args.output, args.dataset_dir, 'train', args.batch_size, True)
    else:
        train_params['data_provider'] = data_provider.get_provider(
            args.task, args.dataset_dir, 'train', args.batch_size, True)

    train_params['predictions'] = partial(
        models.get_model, args.task, train_params['data_provider'].num_outs)
    
    return train_params

if __name__ == '__main__':
    args = parser.parse_args()
    train_params = get_train_params(args)
    e2u.TrainEval(**train_params).start_training()
