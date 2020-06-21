import glob
import os
import pandas as pd
import collections


def prepare_data(task_data_path, transcription_path, class_name, cont_emotions, evaluate_test, predict_partition):

    if predict_partition is not None:
        assert predict_partition in ['train', 'devel','test'], "Partition name not in 'train','devel','test'"
        predict = True 

    id_to_partition, partition_to_id = get_partition(task_data_path)
    data = {}
    
    if predict:
        # prediction only
        segment_txt = []
        ys = []
        files = []
        segments = []
        for sample_id in sorted(partition_to_id[predict_partition]):
            transcription_files = glob.glob(os.path.join(transcription_path, str(sample_id), '*.' + 'csv'))

            for file in sorted(transcription_files, key = sort_trans_files):
                df = pd.read_csv(file, delimiter = ',')
                words = df['word'].tolist()
                files.append(file.split(os.path.sep)[-1].split('.')[0].split('_')[0])
                segments.append(file.split(os.path.sep)[-1].split('.')[0].split('_')[1])   
                segment_txt.append(" ".join(words))
        data[predict_partition] = {'text':segment_txt, 'id':files, 'segment_id':segments}
    else: 
        # training with test labels available
        for partition in partition_to_id.keys():
            segment_txt = []
            ys = []

            if partition != 'test' or (partition == 'test' and evaluate_test):
                for sample_id in sorted(partition_to_id[partition]):
                    transcription_files = glob.glob(os.path.join(transcription_path, str(sample_id), '*.' + 'csv'))

                    for file in sorted(transcription_files, key = sort_trans_files):
                        df = pd.read_csv(file, delimiter = ',')
                        words = df['word'].tolist()   
                        segment_txt.append(" ".join(words))

                        # training without test labels available
                        label_file = os.path.join(task_data_path, 'label_segments', class_name, str(sample_id) + ".csv")
                        if args.class_name in ['arousal', 'valence'] and args.cont_emotions:
                            y_list = read_cont_scores(label_file)
                        else:
                            y_list = read_classification_classes(label_file)

                        for y in y_list:
                            ys.append(y)

            data[partition] = {'text':segment_txt, 'labels':ys}

    return data

def get_partition(task_data_path, path="../../data/processed_tasks/metadata/partition.csv"):

    # any label to collect filenames safely
    names = glob.glob(os.path.join(task_data_path, 'label_segments', 'arousal', '*.' + 'csv'))
    sample_ids = []
    for n in names:
        name_split = n.split(os.path.sep)[-1].split('.')[0]
        sample_ids.append(int(name_split))
    sample_ids = set(sample_ids)

    df = pd.read_csv(path, delimiter=",")
    data = df[["Id", "Proposal"]].values

    id_to_partition = dict()
    partition_to_id = collections.defaultdict(set)

    for i in range(data.shape[0]):
        sample_id = int(data[i, 0])
        partition = data[i, 1]

        if sample_id not in sample_ids:
            continue

        id_to_partition[sample_id] = partition
        partition_to_id[partition].add(sample_id)

    return id_to_partition, partition_to_id

def get_class_names(class_name, path="../../data/processed_tasks/metadata/"):
    if class_name == 'topic':
        df = pd.read_csv(os.path.join(path, 'topic_class_mapping.csv'))
        return df['topic'].values.tolist()
    else:
        df = pd.read_csv(os.path.join(path, 'emotion_class_mapping.csv'))
        return df['emotion'].values.tolist()

def load_id2topic(save_path):
    df = pd.read_csv(save_path)
    df = df.values.tolist()
    id2topic = {row[0]:row[1] for row in df}
    return id2topic

def classid_to_classname(labels, save_path):
    id2topic = load_id2topic(save_path)
    return np.vectorize(id2topic.get)(labels)

def read_classification_classes(label_file):
    df = pd.read_csv(label_file, delimiter=",", usecols=['class_id'])
    y_list =  df['class_id'].tolist()
    return y_list

def read_cont_scores(label_file):
    df = pd.read_csv(label_file, delimiter=",", usecols=['mean'])
    return df['mean'].tolist()

def sort_trans_files(elem):
    return int(elem.split('_')[-1].split('.')[0])
