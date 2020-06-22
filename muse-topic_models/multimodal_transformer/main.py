# Code modified from https://github.com/yaohungt/Multimodal-Transformer/tree/master/src

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"  # specify which GPU(s) to be used
import csv
import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train
import argparse

from config import update_config, save_config, load_config

parser = argparse.ArgumentParser(description='Train Multimodal-Transformer')
# raw data paths
parser.add_argument('-c','--class_name', type=str, dest='class_name', required=True, action='store',
                    default='topic',
                    help='specify which class of task 3 should be predicted')
parser.add_argument('-n','--experiment_name', type=str, dest='experiment_name', required=True, action='store',
                    default='standard',
                    help='specify the name of the experiment')
parser.add_argument('-evaluate_test', '--evaluate_test', dest = 'evaluate_test', required = False, action = 'store_true', 
                    help = 'specify if the model should evaluate on test (assuming labels available).')

parser.add_argument('-a','--aligned', dest='aligned', required=False, action='store_true',
                    help='specify if data is aligned')
parser.add_argument('-vi','--vision', type=str, dest='vision', required=False, action='store',
                    default="xception",
                    help='specify the feature name')
parser.add_argument('-te','--text', type=str, dest='text', required=False, action='store',
                    default="fasttext",
                    help='specify the feature name')
parser.add_argument('-au','--audio', type=str, dest='audio', required=False, action='store',
                    default="egemaps",
                    help='specify the feature name')
parser.add_argument('-ca','--cache', dest='cache', required=False, action='store_true',
                    help='specify if data preprocessing should be stored')

# not used in challenge
parser.add_argument('-er', '--cont_emotions', dest = 'cont_emotions', required = False, action = 'store_true', 
                    help = 'specify if arousal and valence are a continuous score')
parser.add_argument('-rm', '--regression_mode', dest = 'regression_mode', required = False, action = 'store_true', 
                    help = 'specify if arousal and valence are a predicted as score')

args = parser.parse_args()


####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

def load_data_pipeline(params):
    print("Start loading the data....")

    source_filename =  "_".join(['data_dict',args.class_name,params['source_name']])
    source_filename += '.pickle'

    print(source_filename)
    train_data = get_data(params, params['dataset'], source_filename,'train',args.cache)
    valid_data = get_data(params, params['dataset'], source_filename, 'valid',args.cache)
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=params['batch_size'], shuffle=True)
    if args.evaluate_test:
        test_data = get_data(params, params['dataset'], source_filename, 'test',args.cache)
        test_loader = DataLoader(test_data, batch_size=params['batch_size'], shuffle=True)
    else:
        test_data = None
        test_loader = None

    params = update_config(args, params, train_data, valid_data, test_data)

    print('Finish loading the data....')
    if not params['aligned']:
        print("### Note: You are running in unaligned mode.")

    return params, train_loader, valid_loader, test_loader


def main():
    params = load_config(args)
    params, train_loader, valid_loader, test_loader = load_data_pipeline(params)
    save_config(params)
    test_loss = train.initiate(params, train_loader, valid_loader, test_loader)

if __name__ == '__main__':
    main()

