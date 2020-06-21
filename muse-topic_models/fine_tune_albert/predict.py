import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
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
from metrics import read_agg_label_files, score_partition

parser = argparse.ArgumentParser(description = 'Score fine-tuned models of AlBert for MuSe-Topic')
# raw data paths
parser.add_argument('-predict', '--predict_partition', dest = 'predict_partition', type = str, required = False,
                    default = 'test',
                    help = 'specify the partition to be predicted.')
parser.add_argument('-evaluate', '--evaluate', dest = 'evaluate', required = False, action = 'store_true', 
                    help = 'specify if the model should evaluate (assuming labels available).')
parser.add_argument('-pd', '--processed_data_path', type = str, dest = 'processed_data_path', required = False, action = 'store', 
                    default = "../../data/processed_tasks/", 
                    help = 'specify the data folder')
parser.add_argument('--model_type', type = str, dest = 'model_type', required = False, action = 'store', 
                    default = 'albert', 
                    help = 'specify the transformer model')
args = parser.parse_args()

def predict_export(data):

    X = data[args.predict_partition]['text']
    predictions = {}

    for class_name in ['arousal', 'valence','topic']: #

        if class_name in ['arousal', 'valence']:
            class_no = 3
        else:
            class_no = 10

        trained_model_path = os.path.join('experiments/best_model/', class_name + str(False))
        model = ClassificationModel(args.model_type, trained_model_path, num_labels = class_no)
        predictions['prediction_' + class_name], _ = model.predict(X)

    predictions['id'] = data[args.predict_partition]['id']
    predictions['segment_id'] = data[args.predict_partition]['segment_id']

    df = pd.DataFrame.from_dict(predictions)  # , orient='index' .T
    header_names = ['id','segment_id','prediction_arousal', 'prediction_valence', 'prediction_topic']
    df[header_names].to_csv(output_path + args.predict_partition + '.csv', header=header_names, index=False)

def evaluate(output_path, task_data_path):

    y_pred_df = read_agg_label_files(os.path.join(output_path + args.predict_partition + '.csv'))

    score_partition('task2', args.predict_partition, y_pred_df, task_data_path)

if __name__  == "__main__":

    output_path = os.path.join('experiments/predictions/')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    task_data_path = os.path.join(args.processed_data_path, 'c2_muse_topic')
    transcription_path = os.path.join(task_data_path, 'transcription_segments')

    #Param = get_parameters()
    data = prepare_data(task_data_path
                        , transcription_path
                        , None
                        , None
                        , None
                        , args.predict_partition)

    predict_export(data)

    if args.evaluate:
        evaluate(output_path, task_data_path)
