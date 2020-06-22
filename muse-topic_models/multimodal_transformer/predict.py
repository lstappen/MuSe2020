import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
import numpy as np
import pandas as pd
import glob, json, argparse, pickle
import torch
from torch.utils.data import DataLoader
from torch import nn

from src.utils import load_model
from config import load_config_base, check_cuda, dim_criterion
from src.dataset import Multimodal_Datasets
from metrics import read_agg_label_files, score_partition

parser = argparse.ArgumentParser(description='Score Multimodal-Transformer for MuSe-Topic')
# raw data paths
parser.add_argument('-n','--experiment_name', type=str, dest='experiment_name', required=True, action='store',
                    default='standard',
                    help='specify the name of the experiment')

parser.add_argument('-pd', '--processed_data_path', type = str, dest = 'processed_data_path', required = False, action = 'store', 
                    default = "../../data/processed_tasks/", 
                    help = 'specify the data folder')
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

parser.add_argument('-predict', '--predict_partition', dest = 'predict_partition', type = str, required = False,
                    default = 'valid',
                    help = 'specify the partition to be predicted.')
parser.add_argument('-evaluate', '--evaluate', dest = 'evaluate', required = False, action = 'store_true', 
                    help = 'specify if the model should evaluate (assuming labels available).')
args = parser.parse_args()



def predict(params, model, loader, partition, class_name):
    model.eval()
    results = []
    with torch.no_grad():
        for i_batch, (batch_X, _, batch_META) in enumerate(loader):
            sample_ind, text, audio, vision = batch_X
        
            if params['use_cuda']:
                with torch.cuda.device(0):
                    text, audio, vision = text.cuda(), audio.cuda(), vision.cuda()
                    
            batch_size = text.size(0)
                        
            net = nn.DataParallel(model) if batch_size > 10 else model
            preds, _ = net(text, audio, vision)

            preds = preds.view(-1,dim_criterion(class_name, False)[0].get(str.lower(params['dataset'].strip()), 1)
    )

            # Collect the results into dictionary
            results.append(preds)
            
    results = torch.cat(results)

    y_pred = results.cpu().data.numpy().argmax(axis=1)
    return y_pred


def data_file_name():

    source_filename = "_".join([args.vision,args.audio,args.text])
    if args.aligned:
        source_filename += '_data'
    else:
        source_filename += '_data_noalign'
    return "_".join(source_filename.split('_'))

def get_data(args, dataset, source_filename, class_name, split='train', cache = True):
    #alignment = 'a' if args['aligned'] else 'na'
    name = source_filename
    if cache:

        data_path = os.path.join(args['data_path'], dataset + f'_{split}_{name}.dt')
        if not os.path.exists(data_path):
            print(f"  - Creating new {split} data")
            data = Multimodal_Datasets(args['data_path'], source_filename, dataset, split
                                        , args['aligned'], class_name, False)
            torch.save(data, data_path)
        else:
            print(f"  - Found cached {split} data")
            data = torch.load(data_path)
    else:
        data = Multimodal_Datasets(args['data_path'], source_filename, dataset, split
                                    , args['aligned'], class_name, False)

    metadata = meta_data(args['data_path'], source_filename, split)
    return data, metadata

def meta_data(dataset_path, source_filename, split_type):
    dataset_path = os.path.join(dataset_path, source_filename) #.pkl' if if_align else data+'_data_noalign.pkl' )
    dataset = pickle.load(open(dataset_path, 'rb'))
    return { 'id': [int(f.split('.')[0]) for f in dataset[split_type]['meta_id']]
            , 'segment_id': dataset[split_type]['meta_segment_id']}
    

def load_data_pipeline(params, class_name):
    print("Start loading the data....")

    source_name = data_file_name()

    source_filename =  "_".join(['data_dict',class_name,source_name])
    source_filename += '.pickle'

    data, metadata = get_data(params
                                , params['dataset']
                                , source_filename
                                , class_name
                                , args.predict_partition, True)
    loader = DataLoader(data, batch_size=32, shuffle=False)

    return loader, metadata

def predict_export(params):

    predictions = {}
    error_class = []

    for class_name in ['arousal', 'valence','topic']: #

        if class_name in ['arousal', 'valence']:
            class_no = 3
        else:
            class_no = 10

        trained_model_path = os.path.join('experiments/pretrained_model/', class_name, args.experiment_name + '_' + data_file_name())
        try:
            model = load_model(path=trained_model_path, name=args.experiment_name)
            loader, metadata = load_data_pipeline(params, class_name)
            predictions['prediction_' + class_name] = predict(params, model, loader, args.predict_partition, class_name)
            torch.cuda.empty_cache()
        except FileNotFoundError as fnfe:
            print("Model not found: " + str(fnfe))
            print("[WARN!] Set all prediction values for this model to 0")
            error_class.append(class_name)
            continue

    predictions['id'] = metadata['id'] #[args.predict_partition]
    predictions['segment_id'] = metadata['segment_id']

    df = pd.DataFrame.from_dict(predictions)  # , orient='index' .T
    if len(error_class) > 0:
        for ec in error_class:
            df['prediction_' + ec] =  np.nan
        df = df.fillna(0)
        
    header_names = ['id','segment_id','prediction_arousal', 'prediction_valence', 'prediction_topic']
    predict_partition = args.predict_partition.replace('valid','devel')
    df[header_names].to_csv(output_path + predict_partition + '.csv', header=header_names, index=False)

def evaluate(output_path, task_data_path):
    predict_partition = args.predict_partition.replace('valid','devel')
    y_pred_df = read_agg_label_files(os.path.join(output_path + predict_partition + '.csv'))

    score_partition('task2', predict_partition, y_pred_df, task_data_path)
    
if __name__  == "__main__":

    params = load_config_base(args)
    params['use_cuda'] = check_cuda(params)

    output_path = os.path.join('experiments/predictions/')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    predict_export(params)

    task_data_path = os.path.join(args.processed_data_path, 'c2_muse_topic')
    if args.evaluate:
        evaluate(output_path, task_data_path)






