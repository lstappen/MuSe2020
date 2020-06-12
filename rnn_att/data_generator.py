import collections

import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.signal as spsig


from common import make_dirs_safe
from configuration import *


def read_task_2_class_label_file(task_folder, target, segment_id):
    sequence_list = list()

    path = task_folder + "/label_segments" + "/" + target + "/" + repr(segment_id) + ".csv"
    df = pd.read_csv(path, delimiter=",")

    chunk_ids = sorted(set(list(df["segment_id"].values.reshape((-1, )))))

    for chunk_id in chunk_ids:
        sequence = df.loc[df['segment_id'] == chunk_id]
        sequence = sequence["class_id"].values.reshape((1, 1))[0, 0]

        sequence = np.float32(sequence)

        s = 0
        if target == "topic":
            subsequence = np.zeros((10, ), dtype=np.float32)
            subsequence[int(sequence)] = 1.0
        elif target in ["arousal", "valence"]:
            subsequence = np.zeros((3,), dtype=np.float32)
            subsequence[int(sequence)] = 1.0
        else:
            raise ValueError

        sequence_list.append((s, chunk_id, subsequence))

    return sequence_list


def read_sequential_label_file(task_folder, target, segment_id):
    sequence_list = list()

    path = task_folder + "/label_segments" + "/" + target + "/" + repr(segment_id) + ".csv"
    df = pd.read_csv(path, delimiter=",")

    chunk_ids = sorted(set(list(df["segment_id"].values.reshape((-1, )))))

    for chunk_id in chunk_ids:
        sequence = df.loc[df['segment_id'] == chunk_id]
        sequence = sequence["value"].values.reshape((sequence.shape[0], 1)).astype(np.float32)

        num_sub_seq = sequence.size // SEQ_LEN
        if sequence.size % SEQ_LEN > 0:
            num_sub_seq += 1

        for s in range(num_sub_seq):
            start_step = s*SEQ_LEN
            end_step = (s+1)*SEQ_LEN
            if s == num_sub_seq - 1:
                subsequence = sequence[start_step:]
            else:
                subsequence = sequence[start_step:end_step]
            sequence_list.append((s, chunk_id, subsequence))

    return sequence_list


def read_feature_file(task_folder, segment_id, feature_name_list):
    feature_list = collections.defaultdict(list)

    feature_len_dict = dict()

    for feature_name in feature_name_list:
        if feature_name == "lld":
            path = task_folder + "/unaligned/" + feature_name + "/" + repr(segment_id) + ".csv"
        else:
            path = task_folder + "/egemaps_aligned/" + feature_name + "/" + repr(segment_id) + ".csv"
        df = pd.read_csv(path, delimiter=",")
        chunk_ids = sorted(set(list(df["segment_id"].values.reshape((-1,)))))

        feature_len_dict[feature_name] = dict()

        for chunk_id in chunk_ids:
            feature = df.loc[df['segment_id'] == chunk_id]
            feature = feature.values[:, 2:]
            feature = feature.reshape(feature.shape).astype(np.float32)

            feature_len_dict[feature_name][chunk_id] = feature.shape[0]

            if feature_name == "lld":
                feature = spsig.resample_poly(feature, up=1, down=25, axis=0)

                try:
                    egemaps_shape = feature_len_dict["egemaps"][chunk_id]
                except KeyError:
                    continue

                if feature.shape[0] > egemaps_shape:
                    feature = feature[:egemaps_shape, :]
                elif feature.shape[0] < egemaps_shape:
                    feature = np.vstack([feature,
                                         feature[-1, :] * np.ones((egemaps_shape - feature.shape[0],
                                                                   feature.shape[1]),
                                                                  dtype=feature.dtype)])
                feature = feature.astype(np.float32)

            s = 0
            subsequence = feature
            if subsequence.shape[0] > 500:
                subsequence = subsequence[:500, :]

            feature_list[feature_name].append((s, chunk_id, subsequence))

    return feature_list


def read_feature_file_subsegment(task_folder, segment_id, feature_name_list):
    feature_list = collections.defaultdict(list)

    feature_len_dict = dict()

    for feature_name in feature_name_list:
        if feature_name == "lld":
            path = task_folder + "/unaligned/" + feature_name + "/" + repr(segment_id) + ".csv"
        else:
            path = task_folder + "/label_aligned/" + feature_name + "/" + repr(segment_id) + ".csv"

        df = pd.read_csv(path, delimiter=",")
        chunk_ids = sorted(set(list(df["segment_id"].values.reshape((-1,)))))

        feature_len_dict[feature_name] = dict()

        for chunk_id in chunk_ids:
            feature = df.loc[df['segment_id'] == chunk_id]
            feature = feature.values[:, 2:]
            feature = feature.reshape(feature.shape).astype(np.float32)

            feature_len_dict[feature_name][chunk_id] = feature.shape[0]

            if feature_name == "lld":
                feature = spsig.resample_poly(feature, up=1, down=25, axis=0)

                try:
                    egemaps_shape = feature_len_dict["egemaps"][chunk_id]
                except KeyError:
                    continue

                if feature.shape[0] > egemaps_shape:
                    feature = feature[:egemaps_shape, :]
                elif feature.shape[0] < egemaps_shape:
                    feature = np.vstack([feature,
                                         feature[-1, :] * np.ones((egemaps_shape - feature.shape[0],
                                                                   feature.shape[1]),
                                                                  dtype=feature.dtype)])
                feature = feature.astype(np.float32)

            num_sub_seq = feature.shape[0] // SEQ_LEN
            if feature.shape[0] % SEQ_LEN > 0:
                num_sub_seq += 1

            for s in range(num_sub_seq):
                start_step = s * SEQ_LEN
                end_step = (s + 1) * SEQ_LEN
                if s == num_sub_seq - 1:
                    subsequence = feature[start_step:, :]
                else:
                    subsequence = feature[start_step:end_step, :]

                feature_list[feature_name].append((s, chunk_id, subsequence))

    return feature_list


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_sample(task_name, tf_records_folder, task_folder, sample_id, feature_names, partition, are_test_labels_available):
    zip_list = list()
    feature_to_index = dict()
    index_counter = 0

    try:
        if task_name in ["task1", "task3"]:
            features = read_feature_file_subsegment(task_folder, sample_id, feature_names)
        elif task_name == "task2":
            feature_names = [n for n in feature_names if n != "lld"]
            features = read_feature_file(task_folder, sample_id, feature_names)
        else:
            raise ValueError

        for feature_name in feature_names:
            zip_list.append(features[feature_name])
            feature_to_index[feature_name] = index_counter
            index_counter += 1

        targets = dict()

        if task_name == "task1":
            target_names = ["arousal",
                            "valence"]
            if (partition == "test") and (not are_test_labels_available):
                targets["arousal"] = None
                targets["valence"] = None
            else:
                targets["arousal"] = read_sequential_label_file(task_folder, "arousal", sample_id)
                targets["valence"] = read_sequential_label_file(task_folder, "valence", sample_id)
        elif task_name == "task2":
            target_names = ["topic",
                            "arousal",
                            "valence"]
            if (partition == "test") and (not are_test_labels_available):
                targets["topic"] = None
                targets["arousal"] = None
                targets["valence"] = None
            else:
                targets["topic"] = read_task_2_class_label_file(task_folder, "topic", sample_id)
                targets["arousal"] = read_task_2_class_label_file(task_folder, "arousal", sample_id)
                targets["valence"] = read_task_2_class_label_file(task_folder, "valence", sample_id)
        elif task_name == "task3":
            target_names = ["trustworthiness",
                            "arousal",
                            "valence"]
            if (partition == "test") and (not are_test_labels_available):
                targets["trustworthiness"] = None
                targets["arousal"] = None
                targets["valence"] = None
            else:
                targets["trustworthiness"] = read_sequential_label_file(task_folder, "trustworthiness", sample_id)
                targets["arousal"] = read_sequential_label_file(task_folder, "arousal", sample_id)
                targets["valence"] = read_sequential_label_file(task_folder, "valence", sample_id)
        else:
            raise NotImplementedError

        for target_name in target_names:
            if not((partition == "test") and (not are_test_labels_available)):
                zip_list.append(targets[target_name])
                feature_to_index[target_name] = index_counter
                index_counter += 1

    except FileNotFoundError:
        print("File not found:", sample_id)
        return

    for iter_tuple in zip(*zip_list):

        name_to_label = dict()
        if not ((partition == "test") and (not are_test_labels_available)):
            for target_name in targets:
                name_to_label[target_name] = list(iter_tuple[feature_to_index[target_name]])

        name_to_feature = dict()
        for feature_name in feature_names:
            name_to_feature[feature_name] = list(iter_tuple[feature_to_index[feature_name]])

        # Normalise using train partition wide statistics.
        for feature_name in feature_names:
            name_to_feature[feature_name][2] = normalise_features_file(name_to_feature[feature_name][2])

        if task_name in ["task1", "task3"]:
            if not ((partition == "test") and (not are_test_labels_available)):
                for target_name in targets:
                    label, support = pad(name_to_label[target_name][2], SEQ_LEN, np.float32)
                    name_to_label[target_name][2] = label
            else:
                support = np.ones((SEQ_LEN, ), dtype=np.float32)
        if task_name in ["task1", "task3"]:
            for feature_name in feature_names:
                feature, _ = pad(name_to_feature[feature_name][2], SEQ_LEN, np.float32)
                name_to_feature[feature_name][2] = feature
        elif task_name == "task2":
            support = np.ones((name_to_feature["deepspectrum"][2].shape[0], 1), dtype=np.float32)
        else:
            raise ValueError

        writer = tf.io.TFRecordWriter(
            tf_records_folder + "/" + '{}_{}_{}.tfrecords'.format(sample_id,
                                                                  name_to_feature["egemaps"][1],
                                                                  name_to_feature["egemaps"][0]))

        def trim_sequences(data_dict):
            min_len = data_dict["egemaps"][2].shape[0]
            for k, v in data_dict.items():
                d = v[2]
                if d.shape[0] < min_len:
                    min_len = d.shape[0]

            for k, v in data_dict.items():
                d = v[2]
                if d.shape[0] > min_len:
                    pass
                data_dict[k][2] = d[:min_len, :]
            return data_dict

        if not ((partition == "test") and (not are_test_labels_available)):
            for target_name in target_names:
                name_to_label[target_name][2] = np.float32(name_to_label[target_name][2])

        for feature_name in feature_names:
            name_to_feature[feature_name][2] = np.float32(name_to_feature[feature_name][2])

        name_to_feature = trim_sequences(name_to_feature)

        if task_name in ["task1", "task3"]:
            step_zip_list = list()
            index_counter = 0
            attribute_to_index = dict()
            if not ((partition == "test") and (not are_test_labels_available)):
                for target_name in target_names:
                    step_zip_list.append(name_to_label[target_name][2])
                    attribute_to_index[target_name] = index_counter
                    index_counter += 1

            for feature_name in feature_names:
                step_zip_list.append(name_to_feature[feature_name][2])
                attribute_to_index[feature_name] = index_counter
                index_counter += 1

            step_zip_list.append(np.float32(support))
            attribute_to_index["support"] = index_counter
            index_counter += 1

            for i, step_list in enumerate(zip(*step_zip_list)):

                step_counter = name_to_feature["egemaps"][0]
                chunk_id = name_to_feature["egemaps"][1]

                tf_record_dict = {
                    'step_id': _int_feature(np.int64(i)),
                    'subchunk_id': _int_feature(np.int64(step_counter)),
                    'chunk_id': _int_feature(np.int64(chunk_id)),
                    'recording_id': _int_feature(np.int64(sample_id)),
                }

                if not ((partition == "test") and (not are_test_labels_available)):
                    for target_name in target_names:
                        tf_record_dict[target_name] = _bytes_feature(step_list[attribute_to_index[target_name]].tobytes())

                for feature_name in feature_names:
                    tf_record_dict[feature_name] = _bytes_feature(step_list[attribute_to_index[feature_name]].tobytes())

                tf_record_dict["support"] = _bytes_feature(step_list[attribute_to_index["support"]].tobytes())

                # Save tf records.
                example = tf.train.Example(features=tf.train.Features(feature=tf_record_dict))

                writer.write(example.SerializeToString())

        elif task_name == "task2":

            step_counter = name_to_feature["egemaps"][0]
            chunk_id = name_to_feature["egemaps"][1]

            tf_record_dict = {
                'step_id': _int_feature(np.int64(0)),
                'subchunk_id': _int_feature(np.int64(step_counter)),
                'chunk_id': _int_feature(np.int64(chunk_id)),
                'recording_id': _int_feature(np.int64(sample_id)),
            }

            if not ((partition == "test") and (not are_test_labels_available)):
                for target_name in target_names:
                    tf_record_dict[target_name] = _bytes_feature(name_to_label[target_name][2].tobytes())

            for feature_name in feature_names:
                tf_record_dict[feature_name] = _bytes_feature(name_to_feature[feature_name][2].tobytes())

            tf_record_dict["support"] = _bytes_feature(np.float32(support).tobytes())

            example = tf.train.Example(features=tf.train.Features(feature=tf_record_dict))

            writer.write(example.SerializeToString())
        else:
            raise ValueError


def normalise_features_file(features):
    mean_f = np.mean(features, axis=0).reshape((1, features.shape[1])).astype(np.float32)
    std_f = np.std(features, axis=0).reshape((1, features.shape[1])).astype(np.float32)
    std_f[std_f == 0.0] = 1.0

    features = (features - mean_f) / std_f

    return features


def normalise_features_partition(features, feature_name, stats):
    stf_f = stats[feature_name]["standard_deviation"]
    stf_f[stf_f == 0.0] = 1.0

    return (features - stats[feature_name]["mean_value"]) / stf_f


def pad(data, max_length, data_type):
    data_length = data.shape[0]

    new_data = np.zeros(shape=(max_length, data.shape[1]),
                        dtype=data_type)

    new_data[:data_length, :] = data

    support = np.zeros(shape=(max_length, 1),
                       dtype=np.float32)
    support[:data_length, 0] = 1.0

    return new_data, support


def contains_nan(array):
    return np.isnan(array).any()


def are_equal_length(feature_list):
    flag = True
    for f in range(1, len(feature_list)):
        if len(feature_list[f-1]) != len(feature_list[f]):
            flag = False

    return flag


def get_all_union(chunk_id_list_list):
    all_chunk_ids = chunk_id_list_list[0] | chunk_id_list_list[0]

    for f in range(1, len(chunk_id_list_list)):
        all_chunk_ids = all_chunk_ids | chunk_id_list_list[f]

    return all_chunk_ids


def get_all_intersection(chunk_id_list_list):
    all_chunk_ids = chunk_id_list_list[0] & chunk_id_list_list[0]

    for f in range(1, len(chunk_id_list_list)):
        all_chunk_ids = all_chunk_ids & chunk_id_list_list[f]

    return all_chunk_ids


def get_partition(partition_proposal_path, name="third"):
    df = pd.read_csv(partition_proposal_path, delimiter=",")
    data = df[["Id", "Proposal"]].values

    id_to_partition = dict()
    partition_to_id = collections.defaultdict(set)

    for i in range(data.shape[0]):
        sample_id = int(data[i, 0])
        partition = data[i, 1]

        if sample_id in [260, 265, 299]:
            continue

        id_to_partition[sample_id] = partition
        partition_to_id[partition].add(sample_id)

    return id_to_partition, partition_to_id


def main(task_name,
         partition_proposal_path,
         tf_records_folder,
         task_folder,
         feature_names,
         are_test_labels_available):
    make_dirs_safe(tf_records_folder)

    id_to_partition, partition_to_id = get_partition(partition_proposal_path)

    for partition in partition_to_id.keys():
        print("Making tfrecords for", partition, "partition.")

        for sample_id in partition_to_id[partition]:
            serialize_sample(task_name,
                             tf_records_folder,
                             task_folder,
                             sample_id,
                             feature_names,
                             partition,
                             are_test_labels_available)


if __name__ == "__main__":
    TASK_NAME = "task2"
    main(task_name=TASK_NAME,
         partition_proposal_path=PARTITION_PROPOSAL_PATH,
         tf_records_folder=TF_RECORDS_FOLDER[TASK_NAME],
         task_folder=TASK_FOLDER[TASK_NAME],
         feature_names=FEATURE_NAMES,
         are_test_labels_available=False)

    TASK_NAME = "task3"
    main(task_name=TASK_NAME,
         partition_proposal_path=PARTITION_PROPOSAL_PATH,
         tf_records_folder=TF_RECORDS_FOLDER[TASK_NAME],
         task_folder=TASK_FOLDER[TASK_NAME],
         feature_names=FEATURE_NAMES,
         are_test_labels_available=False)

    TASK_NAME = "task1"
    main(task_name=TASK_NAME,
         partition_proposal_path=PARTITION_PROPOSAL_PATH,
         tf_records_folder=TF_RECORDS_FOLDER[TASK_NAME],
         task_folder=TASK_FOLDER[TASK_NAME],
         feature_names=FEATURE_NAMES,
         are_test_labels_available=False)
