#!/usr/bin/env python
# coding: utf-8
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # specify which GPU(s) to be used
import numpy as np
import pandas as pd
import glob, json, argparse
from multiprocessing import cpu_count
from sklearn.utils import class_weight

import torch
torch.cuda.empty_cache()
from simpletransformers.classification import ClassificationModel

from config import get_parameters
from data import prepare_data, get_class_names
from evaluation import regression_results, classification_results, mean_absolute_error, uar, f1, accuracy_score

parser = argparse.ArgumentParser(description = 'Supervised fine-tuning of AlBert for MuSe-Topic')
# raw data paths
parser.add_argument('-c', '--class_name', type = str, dest = 'class_name', required = True, action = 'store', 
                    default = 'topic', 
                    help = 'specify which class of c2_muse_topic should be predicted (arousal, valence, topic)')
parser.add_argument('-pd', '--processed_data_path', type = str, dest = 'processed_data_path', required = False, action = 'store', 
                    default = "../../data/processed_tasks/", 
                    help = 'specify the data folder')
parser.add_argument('-evaluate_test', '--evaluate_test', dest = 'evaluate_test', required = False, action = 'store_true', 
                    help = 'specify if the model should evaluate on test (assuming labels available).')

# static except you want to try out other models
parser.add_argument('--model_type', type = str, dest = 'model_type', required = False, action = 'store', 
                    default = 'albert', 
                    help = 'specify the transformer model')
parser.add_argument('--model_name', type = str, dest = 'model_name', required = False, action = 'store', 
                    default = 'albert-xxlarge-v2', 
                    help = 'specify Transformer model name or path to Transformer model file.')
# not used in challenge
parser.add_argument('-con', '--cont_emotions', dest = 'cont_emotions', required = False, action = 'store_true', 
                    help = 'specify if arousal and valence are continuous (True) or aggregated (False)')
# not used in challenge 
parser.add_argument('-rm', '--regression_mode', dest = 'regression_mode', required = False, action = 'store_true', 
                    help = 'specify if arousal and valence are a score (True) or a class (False)') 
args = parser.parse_args()


def main(Param, data):

    X_train, y_train = data['train']['text'], data['train']['labels']
    X_devel, y_devel = data['devel']['text'], data['devel']['labels']
    print('train - X: {} - y: {}'.format(len(X_train), len(y_train)))
    print('devel - X: {} - y: {}'.format(len(X_devel), len(y_devel)))

    if args.evaluate_test:
        X_test, y_test = data['test']['text'], data['test']['labels']
        print('test  - X: {} - y: {}'.format(len(X_test), len(y_test)))
        return data, X_train, y_train, X_devel, y_devel, X_test, y_test
    else:
        print('test  - X: {} - y: not loaded'.format(len(ata['test']['text'])))
        del data['test']['labels']
        return data, X_train, y_train, X_devel, y_devel, data['test']['text']

    y_names = get_class_names(args.class_name)

    train_df = pd.DataFrame.from_dict(data['train'])
    devel_df = pd.DataFrame.from_dict(data['devel'])
    test_df = pd.DataFrame.from_dict(data['test'])

    #args.regression_mode and 
    if args.class_name in ['arousal', 'valence'] and args.regression_mode:
        # create a RegressionModel
        Param['regression'] = True
        model = ClassificationModel(args.model_type, args.model_name, num_labels = 1, args = Param)
        model.train_model(train_df, eval_df = devel_df, mae = mean_absolute_error)

        # load best model
        del model
        model = ClassificationModel(args.model_type, Param['best_model_dir'], num_labels = 1)
        regression_results(Param, model, train_df, y_train, y_names, 'train')
        regression_results(Param, model, devel_df, y_devel, y_names, 'devel')
        if args.evaluate_test:
            regression_results(Param, model, test_df, y_test, y_names, 'test')

    else:
        # create a ClassificationModel
        # weights to counteract label imbalance
        weights_list = list(class_weight.compute_class_weight('balanced', np.unique(y_train), y_train))
        if args.class_name in ['arousal', 'valence']:
            class_no = 3#6
        else:
            class_no = 10

        weights_dict = dict(zip(range(class_no), weights_list))
        weights = [weights_dict[label] for label in y_train]

        model = ClassificationModel(args.model_type, args.model_name, num_labels = class_no, weight = weights_list, args = Param)
        model.train_model(train_df, eval_df = devel_df, uar = uar, f1 = f1, acc = accuracy_score)

        # load best model
        del model
        model = ClassificationModel(args.model_type, Param['best_model_dir'], num_labels = class_no)

        classification_results(Param, model, train_df, y_train, y_names, 'train', args.class_name)
        classification_results(Param, model, devel_df, y_devel, y_names, 'devel', args.class_name)
        if args.evaluate_test:
            classification_results(Param, model, test_df, y_test, y_names, 'test', args.class_name)
        else:
            print("Use predict.py to create test predictions")

    # to be save - free memory
    del model
    torch.cuda.empty_cache()


if __name__  == "__main__":

    task_data_path = os.path.join(args.processed_data_path, 'c2_muse_topic')
    transcription_path = os.path.join(task_data_path, 'transcription_segments')

    Param = get_parameters()

    # create working folders
    if not os.path.exists(Param['output_dir']):
        os.makedirs(Param['output_dir'])
    if not os.path.exists(Param['cache_dir']):
        os.makedirs(Param['cache_dir'])
    if not os.path.exists(Param['best_model_dir']):
        os.makedirs(Param['best_model_dir'])
    with open(os.path.join(Param['output_dir'], 'parameter.json'), 'w') as pa:
        json.dump(Param, pa, indent = ' ')

    data = prepare_data(task_data_path
                        , transcription_path
                        , args.class_name
                        , args.cont_emotions
                        , args.evaluate_test
                        , None)

    if not args.predict_test:
        main(Param, task_data_path, transcription_path)