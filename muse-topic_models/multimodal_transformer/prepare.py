import pandas as pd
import numpy as np
import pickle
import os
from os import listdir
from os.path import isfile, join
import argparse

parser = argparse.ArgumentParser(description='Prepare data for multimodal transformer.')
parser.add_argument('-c','--class_name', type=str, dest='class_name', required=True, action='store',
                    default='topic',
                    help='specify which class of task 2 should be predicted')
parser.add_argument('-a','--aligned', dest='aligned', required=False, action='store_true',
                    help='specify if data are aligned')
parser.add_argument('-at','--alignment_type', type=str, dest='alignment_type', required=False, action='store',
                    default="unaligned", help='specify the alignment') #TODO: fuse with -a
# data paths
parser.add_argument('-cf','--data_dir', type=str, dest='data_dir', required=False, action='store',
                    default="../../data/processed_tasks/c2_muse_topic/",
                    help='specify the task folder')
parser.add_argument('-vi','--vision_path', type=str, dest='vision_path', required=False, action='store',
                    default="../../data/processed_tasks/c2_muse_topic/feature_segments/unaligned/xception",
                    help='specify the vision folder')
parser.add_argument('-te','--text_path', type=str, dest='text_path', required=False, action='store',
                    default="../../data/processed_tasks/c2_muse_topic/feature_segments/unaligned/fasttext",
                    help='specify the text folder')
parser.add_argument('-au','--audio_path', type=str, dest='audio_path', required=False, action='store',
                    default="../../data/processed_tasks/c2_muse_topic/feature_segments/unaligned/egemaps",
                    help='specify the audio folder')
parser.add_argument('-out','--output_path', dest='output_path', required=False,
                    default="./data",
                    help='specify the output directory')
# modes
parser.add_argument('-exa','--example_mode', dest='example_mode', required=False, action='store_true',
                    help='specify if test mode is activated (only few test data)')
parser.add_argument('-et', '--evaluate_test', dest = 'evaluate_test', required = False, action = 'store_true', 
                    help = 'specify if the model should run and evaluate on test (assuming labels available).')

# not used in challenge
parser.add_argument('-er','--cont_emotions', dest='cont_emotions', required=False, action='store_true',
                    help='specify if arousal and valence are a continuous score')

args = parser.parse_args()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

# replace unaligned in path
vision_path = args.vision_path.replace('unaligned',args.alignment_type )
text_path = args.text_path.replace('unaligned',args.alignment_type )
audio_path = args.audio_path.replace('unaligned',args.alignment_type )

label_path = join(args.data_dir, "label_segments",args.class_name)

feature_config = {'fasttext' : {'feature_no':300,'feature_type':'text'},
                'egemaps' : {'feature_no':88,'feature_type':'audio'},
                'deepspectrum' : {'feature_no':4096,'feature_type':'audio'},
                'xception' : {'feature_no':2048,'feature_type':'vision'},
                'gocar' : {'feature_no':350,'feature_type':'vision'},
                'au' : {'feature_no':35,'feature_type':'vision'},
                'gaze' : {'feature_no':288,'feature_type':'vision'},
                'landmarks_2d' : {'feature_no':136,'feature_type':'vision'},
                'landmarks_3d' : {'feature_no':204,'feature_type':'vision'},
                'pqm' : {'feature_no':40,'feature_type':'vision'},
                'pose' : {'feature_no':6,'feature_type':'vision'},
                'openpose' : {'feature_no':54,'feature_type':'vision'},
                'vggface' : {'feature_no':512,'feature_type':'vision'}
}

VISION = args.vision_path.split(os.path.sep)[-1]
AUDIO = args.audio_path.split(os.path.sep)[-1]
TEXT = args.text_path.split(os.path.sep)[-1]

if args.aligned:
    LENGTH = 220
    VISION_SHAPE = (LENGTH, feature_config[VISION]['feature_no'])
    TEXT_SHAPE = (LENGTH, feature_config[TEXT]['feature_no'])
    AUDIO_SHAPE = (LENGTH, feature_config[AUDIO]['feature_no']) 
else:
    VISION_SHAPE = (220, feature_config[VISION]['feature_no'])
    AUDIO_SHAPE = (220, feature_config[AUDIO]['feature_no'])
    if feature_config[TEXT]['feature_type'] =='text':
        TEXT_SHAPE = (170, feature_config[TEXT]['feature_no'])
    else:
        TEXT_SHAPE = (220, feature_config[TEXT]['feature_no'])

all_ids = [f for f in listdir(label_path) if isfile(join(label_path, f))]

def load_partition(mode, path="../../data/raw/metadata/partition.csv"):

    df = pd.read_csv(path)
    df = df[df["Proposal"] == mode]
    videos = df['Id'].tolist()
    videos = [str(v) + ".csv" for v in videos]
    return videos

def load_data(data, mode):
    vid_ids = load_partition(mode)
    # for test purposes:
    if args.example_mode:
        vid_ids = vid_ids[:4]

    for file in vid_ids:
        if not (isfile(join(label_path, file)) and isfile(join(vision_path, file)) \
                and isfile(join(audio_path, file)) and isfile(join(text_path, file))):
            continue
        labels_df = pd.read_csv(join(label_path, file)) #if test than masked
        vision_df = pd.read_csv(join(vision_path, file))
        text_df = pd.read_csv(join(text_path, file))
        audio_df = pd.read_csv(join(audio_path, file))
        data = extend_data_dict(data, labels_df, vision_df, text_df, audio_df, file)

    return data

def extend_data_dict(data_dict, labels_df, vision_df, text_df, audio_df, file):
    def preprocess_relevant_part(df, segment_id, drop_columns = 
                            ['timestamp', 'segment_id']):
        df = df[df['segment_id'] == segment_id]
        relevant_df = df.drop(columns=drop_columns)
        values = relevant_df.values.tolist() # note: this can be empty.
        return values

    if args.alignment_type == 'fasttext_aligned':
        drop_columns = ['start','end', 'segment_id']
    elif args.alignment_type =='unaligned':
        drop_columns = ['timestamp', 'segment_id']

    for i, row in labels_df.iterrows():
        segment_id = row['segment_id']

        if args.class_name == 'topic':
            data_dict['labels'].append(row['class_id'])
        elif args.class_name in ['arousal','valence']:
            if args.cont_emotions:
                data_dict['labels'].append(row['mean'])
            else:
                data_dict['labels'].append(row['class_id'])

        audio = preprocess_relevant_part(audio_df, segment_id, drop_columns)
        data_dict['audio'].append(audio)
    
        vision = preprocess_relevant_part(vision_df, segment_id , drop_columns)
        data_dict['vision'].append(vision)

        # TO-DO: run experiments without text. Could be implemented in a more generic way
        if feature_config[TEXT]['feature_type'] == 'text':
            text = preprocess_relevant_part(text_df, segment_id, [
                    'start', 'end', 'segment_id'])
        else:
             text = preprocess_relevant_part(text_df, segment_id, drop_columns)           
        data_dict['text'].append(text)

        data_dict['meta_segment_id'].append(segment_id)
        data_dict['meta_id'].append(file)

    return data_dict


def format_data(data_dict, mode):
    feature_keys = ["vision", "audio", "text"]
    feature_shapes = [VISION_SHAPE, AUDIO_SHAPE, TEXT_SHAPE]

    #if feature_keys == 'audio':
    for key, shape in zip(feature_keys, feature_shapes):
        features = data_dict[key]

        # truncate:
        features = [f[:shape[0]] for f in features]

        # pad: 
        features = [f + [[0]*shape[1]]*(shape[0]-len(f)) for f in features]
        
        features = np.array(features)
        data_dict[key] = features

    if mode != 'test' or args.evaluate_test:

        # handle labels seperately:
        labels = data_dict["labels"]

        if args.class_name == 'topic':
            # Transformation not necessary anymore
            # labels = [get_id(label) for label in labels]
            labels = np.array(labels).astype(int)
            num_classes = 10
            labels = labels.reshape(-1)
            one_hot_labels = np.eye(num_classes)[labels]
            one_hot_labels = np.reshape(one_hot_labels, (-1,1,num_classes))
            data_dict["labels"] = one_hot_labels
        elif args.class_name in ['arousal','valence']:
            if args.cont_emotions:
                labels = np.array(labels)
                data_dict['labels'] = np.reshape(labels, (-1,1)) 
            else:
                labels = np.array(labels).astype(int)
                num_classes = 3
                labels = labels.reshape(-1)
                one_hot_labels = np.eye(num_classes)[labels]
                one_hot_labels = np.reshape(one_hot_labels, (-1,1,num_classes))
                data_dict["labels"] = one_hot_labels
    else:
        labels = data_dict["labels"]
        if args.class_name == 'topic':
            num_classes = 10
        elif args.class_name in ['arousal','valence']:
            num_classes = 3
        #0    
        data_dict["labels"] = np.reshape(np.zeros((num_classes,len(labels))), (-1,1,num_classes))

        if  args.cont_emotions:
            data_dict['labels'] = np.reshape(np.zeros((1,len(labels))), (-1,1)) 

    return data_dict


def save_data_dict(data, path=None):

    if path is None:
        path = os.path.join(args.output_path, "_".join(['data_dict',args.class_name,VISION,AUDIO,TEXT]))
        if args.aligned:
            path += '_data'
        else:
            path += '_data_noalign'
        if args.cont_emotions:
            path += '_cont'
        path += '.pickle'

    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data_dict(path=None):
    if path is None:
        path = "_".join(['data_dict',args.class_name,VISION,AUDIO,TEXT])
        if args.aligned:
            path += '_data'
        else:
            path += '_data_noalign'
        if args.cont_emotions:
            path += '_cont'
        path += '.pickle'

    with open(path, 'rb') as handle:
        return pickle.load(handle)


def init_data():
    return {'labels':[], 'vision':[], 'text':[], 'audio':[], 'meta_segment_id': [], 'meta_id': []}

def prepare_data(mode):
    data = init_data()
    data = load_data(data, mode)
    data = format_data(data, mode)
    return data

def prepare():
    data = {}
    for key in ["train", "test", "devel"]:
        data[key] = prepare_data(key)
    data['valid'] = data.pop('devel') # rename key
    return data


if __name__ == "__main__":
    if args.cont_emotions:
        print("Continuous mode is switched on for emotion labels")
    data = prepare()
    save_data_dict(data)
    print(data.keys())
    for key in data.keys():
        for f in data[key].keys():
            if 'meta' in f:
                print("{} {} -> {}".format(key, f, len(data[key][f])))
            else:
                print("{} {} -> {}".format(key, f, data[key][f].shape))
