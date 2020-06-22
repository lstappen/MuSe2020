import torch
import os
import pandas as pd
from src.dataset import Multimodal_Datasets


def get_data(args, dataset, source_filename, split='train', cache = True):
    #alignment = 'a' if args['aligned'] else 'na'
    name = args['source_name']
    if cache:
        data_path = os.path.join(args['data_path'], dataset) + f'_{split}_{name}.dt'
        if not os.path.exists(data_path):
            print(f"  - Creating new {split} data")
            data = Multimodal_Datasets(args['data_path'], source_filename, dataset, split
                                        , args['aligned'], args['class_name'], args['cont_emotions'])
            torch.save(data, data_path)
        else:
            print(f"  - Found cached {split} data")
            data = torch.load(data_path)
    else:
        data = Multimodal_Datasets(args['data_path'], source_filename, dataset, split
                                    , args['aligned'], args['class_name'], args['cont_emotions'])
    return data


def save_model(path, model, name):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model, os.path.join(path,f'{name}.pt'))


def load_model(path, name):
    model = torch.load(os.path.join(path,f'{name}.pt'))
    return model

def get_class_names(class_name, path="../../data/raw/metadata/"):
    if class_name == 'topic':
        df = pd.read_csv(os.path.join(path, 'topic_class_mapping.csv'))
        return df['topic'].values.tolist()
    else:
        df = pd.read_csv(os.path.join(path, 'emotion_class_mapping.csv'))
        return df['emotion'].values.tolist()