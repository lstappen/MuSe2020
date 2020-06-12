from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path

import tensorflow as tf

from configuration import *


def get_split(dataset_dir,
              is_training,
              task_name,
              split_name,
              are_test_labels_available,
              id_to_partition,
              feature_names,
              batch_size,
              seq_length,
              buffer_size):
    root_path = Path(dataset_dir)
    paths_unfiltered = [str(x) for x in root_path.glob('*.tfrecords')]

    paths = list()
    for path in paths_unfiltered:
        path_split = path.split("/")[-1]
        path_split = path_split.split(".")[0]
        path_split = path_split.split("_")[0]
        sample_id = int(path_split)
        if id_to_partition[sample_id] == split_name:
            paths.append(path)

    dataset = tf.data.TFRecordDataset(paths)

    if not ((split_name == "test") and (not are_test_labels_available)):
        if task_name == "task1":

            dataset = dataset.map(lambda x: tf.parse_single_example(x,
                                                                    features=
                                                                    {
                                                                        'step_id': tf.FixedLenFeature([], tf.int64),
                                                                        'chunk_id': tf.FixedLenFeature([], tf.int64),
                                                                        'recording_id': tf.FixedLenFeature([],
                                                                                                           tf.int64),
                                                                        'au': tf.FixedLenFeature([], tf.string),
                                                                        'deepspectrum': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'egemaps': tf.FixedLenFeature([], tf.string),
                                                                        'fasttext': tf.FixedLenFeature([], tf.string),
                                                                        'gaze': tf.FixedLenFeature([], tf.string),
                                                                        'gocar': tf.FixedLenFeature([], tf.string),
                                                                        'landmarks_2d': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'landmarks_3d': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'lld': tf.FixedLenFeature([], tf.string),
                                                                        'openpose': tf.FixedLenFeature([], tf.string),
                                                                        'pdm': tf.FixedLenFeature([], tf.string),
                                                                        'pose': tf.FixedLenFeature([], tf.string),
                                                                        'vggface': tf.FixedLenFeature([], tf.string),
                                                                        'xception': tf.FixedLenFeature([], tf.string),
                                                                        'support': tf.FixedLenFeature([], tf.string),
                                                                        'arousal': tf.FixedLenFeature([], tf.string),
                                                                        'valence': tf.FixedLenFeature([], tf.string),
                                                                    }
                                                                    ))

            dataset = dataset.map(lambda x:
                                  {
                                      'step_id': tf.cast(tf.reshape(x['step_id'], (1,)), tf.int32),
                                      'chunk_id': tf.cast(tf.reshape(x['chunk_id'], (1,)), tf.int32),
                                      'recording_id': tf.cast(tf.reshape(x['recording_id'], (1,)), tf.int32),
                                      'au': tf.reshape(tf.decode_raw(x['au'], tf.float32), (FEATURE_NUM["au"],)),
                                      'deepspectrum': tf.reshape(tf.decode_raw(x['deepspectrum'], tf.float32),
                                                                 (FEATURE_NUM["deepspectrum"],)),
                                      'egemaps': tf.reshape(tf.decode_raw(x['egemaps'], tf.float32),
                                                            (FEATURE_NUM["egemaps"],)),
                                      'fasttext': tf.reshape(tf.decode_raw(x['fasttext'], tf.float32),
                                                             (FEATURE_NUM["fasttext"],)),
                                      'gaze': tf.reshape(tf.decode_raw(x['gaze'], tf.float32), (FEATURE_NUM["gaze"],)),
                                      'gocar': tf.reshape(tf.decode_raw(x['gocar'], tf.float32),
                                                          (FEATURE_NUM["gocar"],)),
                                      'landmarks_2d': tf.reshape(tf.decode_raw(x['landmarks_2d'], tf.float32),
                                                                 (FEATURE_NUM["landmarks_2d"],)),
                                      'landmarks_3d': tf.reshape(tf.decode_raw(x['landmarks_3d'], tf.float32),
                                                                 (FEATURE_NUM["landmarks_3d"],)),
                                      'lld': tf.reshape(tf.decode_raw(x['lld'], tf.float32), (FEATURE_NUM["lld"],)),
                                      'openpose': tf.reshape(tf.decode_raw(x['openpose'], tf.float32),
                                                             (FEATURE_NUM["openpose"],)),
                                      'pdm': tf.reshape(tf.decode_raw(x['pdm'], tf.float32), (FEATURE_NUM["pdm"],)),
                                      'pose': tf.reshape(tf.decode_raw(x['pose'], tf.float32), (FEATURE_NUM["pose"],)),
                                      'vggface': tf.reshape(tf.decode_raw(x['vggface'], tf.float32),
                                                            (FEATURE_NUM["vggface"],)),
                                      'xception': tf.reshape(tf.decode_raw(x['xception'], tf.float32),
                                                             (FEATURE_NUM["xception"],)),
                                      'support': tf.reshape(tf.decode_raw(x['support'], tf.float32), (1,)),
                                      'arousal': tf.reshape(tf.decode_raw(x['arousal'], tf.float32), (1,)),
                                      'valence': tf.reshape(tf.decode_raw(x['valence'], tf.float32), (1,)),
                                  }
                                  )
        elif task_name == "task2":

            dataset = dataset.map(lambda x: tf.parse_single_example(x,
                                                                    features=
                                                                    {
                                                                        'step_id': tf.FixedLenFeature([], tf.int64),
                                                                        'chunk_id': tf.FixedLenFeature([], tf.int64),
                                                                        'recording_id': tf.FixedLenFeature([],
                                                                                                           tf.int64),
                                                                        'au': tf.FixedLenFeature([], tf.string),
                                                                        'deepspectrum': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'egemaps': tf.FixedLenFeature([], tf.string),
                                                                        'fasttext': tf.FixedLenFeature([], tf.string),
                                                                        'gaze': tf.FixedLenFeature([], tf.string),
                                                                        'gocar': tf.FixedLenFeature([], tf.string),
                                                                        'landmarks_2d': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'landmarks_3d': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'openpose': tf.FixedLenFeature([], tf.string),
                                                                        'pdm': tf.FixedLenFeature([], tf.string),
                                                                        'pose': tf.FixedLenFeature([], tf.string),
                                                                        'vggface': tf.FixedLenFeature([], tf.string),
                                                                        'xception': tf.FixedLenFeature([], tf.string),
                                                                        'support': tf.FixedLenFeature([], tf.string),
                                                                        'topic': tf.FixedLenFeature([], tf.string),
                                                                        'arousal': tf.FixedLenFeature([], tf.string),
                                                                        'valence': tf.FixedLenFeature([], tf.string),
                                                                    }
                                                                    ))

            dataset = dataset.map(lambda x:
                                  {
                                      'step_id': tf.cast(tf.reshape(x['step_id'], (1,)), tf.int32),
                                      'chunk_id': tf.cast(tf.reshape(x['chunk_id'], (1,)), tf.int32),
                                      'recording_id': tf.cast(tf.reshape(x['recording_id'], (1,)), tf.int32),
                                      'au': tf.reshape(tf.decode_raw(x['au'], tf.float32),
                                                       (-1, FEATURE_NUM["au"],)),
                                      'deepspectrum': tf.reshape(tf.decode_raw(x['deepspectrum'], tf.float32),
                                                                 (-1, FEATURE_NUM["deepspectrum"],)),
                                      'egemaps': tf.reshape(tf.decode_raw(x['egemaps'], tf.float32),
                                                            (-1, FEATURE_NUM["egemaps"],)),
                                      'fasttext': tf.reshape(tf.decode_raw(x['fasttext'], tf.float32),
                                                             (-1, FEATURE_NUM["fasttext"],)),
                                      'gaze': tf.reshape(tf.decode_raw(x['gaze'], tf.float32),
                                                         (-1, FEATURE_NUM["gaze"],)),
                                      'gocar': tf.reshape(tf.decode_raw(x['gocar'], tf.float32),
                                                          (-1, FEATURE_NUM["gocar"],)),
                                      'landmarks_2d': tf.reshape(tf.decode_raw(x['landmarks_2d'], tf.float32),
                                                                 (-1, FEATURE_NUM["landmarks_2d"],)),
                                      'landmarks_3d': tf.reshape(tf.decode_raw(x['landmarks_3d'], tf.float32),
                                                                 (-1, FEATURE_NUM["landmarks_3d"],)),
                                      'openpose': tf.reshape(tf.decode_raw(x['openpose'], tf.float32),
                                                             (-1, FEATURE_NUM["openpose"],)),
                                      'pdm': tf.reshape(tf.decode_raw(x['pdm'], tf.float32), (-1, FEATURE_NUM["pdm"],)),
                                      'pose': tf.reshape(tf.decode_raw(x['pose'], tf.float32),
                                                         (-1, FEATURE_NUM["pose"],)),
                                      'vggface': tf.reshape(tf.decode_raw(x['vggface'], tf.float32),
                                                            (-1, FEATURE_NUM["vggface"],)),
                                      'xception': tf.reshape(tf.decode_raw(x['xception'], tf.float32),
                                                             (-1, FEATURE_NUM["xception"],)),
                                      'support': tf.reshape(tf.decode_raw(x['support'], tf.float32), (-1, 1,)),
                                      'topic': tf.reshape(tf.decode_raw(x['topic'], tf.float32), (10,)),
                                      'arousal': tf.reshape(tf.decode_raw(x['arousal'], tf.float32), (3,)),
                                      'valence': tf.reshape(tf.decode_raw(x['valence'], tf.float32), (3,)),
                                  }
                                  )
        elif task_name == "task3":
            dataset = dataset.map(lambda x: tf.parse_single_example(x,
                                                                    features=
                                                                    {
                                                                        'step_id': tf.FixedLenFeature([], tf.int64),
                                                                        'chunk_id': tf.FixedLenFeature([], tf.int64),
                                                                        'recording_id': tf.FixedLenFeature([],
                                                                                                           tf.int64),
                                                                        'au': tf.FixedLenFeature([], tf.string),
                                                                        'deepspectrum': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'egemaps': tf.FixedLenFeature([], tf.string),
                                                                        'fasttext': tf.FixedLenFeature([], tf.string),
                                                                        'gaze': tf.FixedLenFeature([], tf.string),
                                                                        'gocar': tf.FixedLenFeature([], tf.string),
                                                                        'landmarks_2d': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'landmarks_3d': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'lld': tf.FixedLenFeature([], tf.string),
                                                                        'openpose': tf.FixedLenFeature([], tf.string),
                                                                        'pdm': tf.FixedLenFeature([], tf.string),
                                                                        'pose': tf.FixedLenFeature([], tf.string),
                                                                        'vggface': tf.FixedLenFeature([], tf.string),
                                                                        'xception': tf.FixedLenFeature([], tf.string),
                                                                        'support': tf.FixedLenFeature([], tf.string),
                                                                        'trustworthiness': tf.FixedLenFeature([],
                                                                                                              tf.string),
                                                                        'arousal': tf.FixedLenFeature([], tf.string),
                                                                        'valence': tf.FixedLenFeature([], tf.string),
                                                                    }
                                                                    ))

            dataset = dataset.map(lambda x:
                                  {
                                      'step_id': tf.cast(tf.reshape(x['step_id'], (1,)), tf.int32),
                                      'chunk_id': tf.cast(tf.reshape(x['chunk_id'], (1,)), tf.int32),
                                      'recording_id': tf.cast(tf.reshape(x['recording_id'], (1,)), tf.int32),
                                      'au': tf.reshape(tf.decode_raw(x['au'], tf.float32), (FEATURE_NUM["au"],)),
                                      'deepspectrum': tf.reshape(tf.decode_raw(x['deepspectrum'], tf.float32),
                                                                 (FEATURE_NUM["deepspectrum"],)),
                                      'egemaps': tf.reshape(tf.decode_raw(x['egemaps'], tf.float32),
                                                            (FEATURE_NUM["egemaps"],)),
                                      'fasttext': tf.reshape(tf.decode_raw(x['fasttext'], tf.float32),
                                                             (FEATURE_NUM["fasttext"],)),
                                      'gaze': tf.reshape(tf.decode_raw(x['gaze'], tf.float32), (FEATURE_NUM["gaze"],)),
                                      'gocar': tf.reshape(tf.decode_raw(x['gocar'], tf.float32),
                                                          (FEATURE_NUM["gocar"],)),
                                      'landmarks_2d': tf.reshape(tf.decode_raw(x['landmarks_2d'], tf.float32),
                                                                 (FEATURE_NUM["landmarks_2d"],)),
                                      'landmarks_3d': tf.reshape(tf.decode_raw(x['landmarks_3d'], tf.float32),
                                                                 (FEATURE_NUM["landmarks_3d"],)),
                                      'lld': tf.reshape(tf.decode_raw(x['lld'], tf.float32), (FEATURE_NUM["lld"],)),
                                      'openpose': tf.reshape(tf.decode_raw(x['openpose'], tf.float32),
                                                             (FEATURE_NUM["openpose"],)),
                                      'pdm': tf.reshape(tf.decode_raw(x['pdm'], tf.float32), (FEATURE_NUM["pdm"],)),
                                      'pose': tf.reshape(tf.decode_raw(x['pose'], tf.float32), (FEATURE_NUM["pose"],)),
                                      'vggface': tf.reshape(tf.decode_raw(x['vggface'], tf.float32),
                                                            (FEATURE_NUM["vggface"],)),
                                      'xception': tf.reshape(tf.decode_raw(x['xception'], tf.float32),
                                                             (FEATURE_NUM["xception"],)),
                                      'support': tf.reshape(tf.decode_raw(x['support'], tf.float32), (1,)),
                                      'trustworthiness': tf.reshape(tf.decode_raw(x['trustworthiness'], tf.float32),
                                                                    (1,)),
                                      'arousal': tf.reshape(tf.decode_raw(x['arousal'], tf.float32), (1,)),
                                      'valence': tf.reshape(tf.decode_raw(x['valence'], tf.float32), (1,)),
                                  }
                                  )
        else:
            raise NotImplementedError
    else:
        if task_name == "task1":

            dataset = dataset.map(lambda x: tf.parse_single_example(x,
                                                                    features=
                                                                    {
                                                                        'step_id': tf.FixedLenFeature([], tf.int64),
                                                                        'chunk_id': tf.FixedLenFeature([], tf.int64),
                                                                        'recording_id': tf.FixedLenFeature([],
                                                                                                           tf.int64),
                                                                        'au': tf.FixedLenFeature([], tf.string),
                                                                        'deepspectrum': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'egemaps': tf.FixedLenFeature([], tf.string),
                                                                        'fasttext': tf.FixedLenFeature([], tf.string),
                                                                        'gaze': tf.FixedLenFeature([], tf.string),
                                                                        'gocar': tf.FixedLenFeature([], tf.string),
                                                                        'landmarks_2d': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'landmarks_3d': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'lld': tf.FixedLenFeature([], tf.string),
                                                                        'openpose': tf.FixedLenFeature([], tf.string),
                                                                        'pdm': tf.FixedLenFeature([], tf.string),
                                                                        'pose': tf.FixedLenFeature([], tf.string),
                                                                        'vggface': tf.FixedLenFeature([], tf.string),
                                                                        'xception': tf.FixedLenFeature([], tf.string),
                                                                        'support': tf.FixedLenFeature([], tf.string),
                                                                    }
                                                                    ))

            dataset = dataset.map(lambda x:
                                  {
                                      'step_id': tf.cast(tf.reshape(x['step_id'], (1,)), tf.int32),
                                      'chunk_id': tf.cast(tf.reshape(x['chunk_id'], (1,)), tf.int32),
                                      'recording_id': tf.cast(tf.reshape(x['recording_id'], (1,)), tf.int32),
                                      'au': tf.reshape(tf.decode_raw(x['au'], tf.float32), (FEATURE_NUM["au"],)),
                                      'deepspectrum': tf.reshape(tf.decode_raw(x['deepspectrum'], tf.float32),
                                                                 (FEATURE_NUM["deepspectrum"],)),
                                      'egemaps': tf.reshape(tf.decode_raw(x['egemaps'], tf.float32),
                                                            (FEATURE_NUM["egemaps"],)),
                                      'fasttext': tf.reshape(tf.decode_raw(x['fasttext'], tf.float32),
                                                             (FEATURE_NUM["fasttext"],)),
                                      'gaze': tf.reshape(tf.decode_raw(x['gaze'], tf.float32), (FEATURE_NUM["gaze"],)),
                                      'gocar': tf.reshape(tf.decode_raw(x['gocar'], tf.float32),
                                                          (FEATURE_NUM["gocar"],)),
                                      'landmarks_2d': tf.reshape(tf.decode_raw(x['landmarks_2d'], tf.float32),
                                                                 (FEATURE_NUM["landmarks_2d"],)),
                                      'landmarks_3d': tf.reshape(tf.decode_raw(x['landmarks_3d'], tf.float32),
                                                                 (FEATURE_NUM["landmarks_3d"],)),
                                      'lld': tf.reshape(tf.decode_raw(x['lld'], tf.float32), (FEATURE_NUM["lld"],)),
                                      'openpose': tf.reshape(tf.decode_raw(x['openpose'], tf.float32),
                                                             (FEATURE_NUM["openpose"],)),
                                      'pdm': tf.reshape(tf.decode_raw(x['pdm'], tf.float32), (FEATURE_NUM["pdm"],)),
                                      'pose': tf.reshape(tf.decode_raw(x['pose'], tf.float32), (FEATURE_NUM["pose"],)),
                                      'vggface': tf.reshape(tf.decode_raw(x['vggface'], tf.float32),
                                                            (FEATURE_NUM["vggface"],)),
                                      'xception': tf.reshape(tf.decode_raw(x['xception'], tf.float32),
                                                             (FEATURE_NUM["xception"],)),
                                      'support': tf.reshape(tf.decode_raw(x['support'], tf.float32), (1,)),
                                  }
                                  )
        elif task_name == "task2":

            dataset = dataset.map(lambda x: tf.parse_single_example(x,
                                                                    features=
                                                                    {
                                                                        'step_id': tf.FixedLenFeature([], tf.int64),
                                                                        'chunk_id': tf.FixedLenFeature([], tf.int64),
                                                                        'recording_id': tf.FixedLenFeature([],
                                                                                                           tf.int64),
                                                                        'au': tf.FixedLenFeature([], tf.string),
                                                                        'deepspectrum': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'egemaps': tf.FixedLenFeature([], tf.string),
                                                                        'fasttext': tf.FixedLenFeature([], tf.string),
                                                                        'gaze': tf.FixedLenFeature([], tf.string),
                                                                        'gocar': tf.FixedLenFeature([], tf.string),
                                                                        'landmarks_2d': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'landmarks_3d': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'openpose': tf.FixedLenFeature([], tf.string),
                                                                        'pdm': tf.FixedLenFeature([], tf.string),
                                                                        'pose': tf.FixedLenFeature([], tf.string),
                                                                        'vggface': tf.FixedLenFeature([], tf.string),
                                                                        'xception': tf.FixedLenFeature([], tf.string),
                                                                        'support': tf.FixedLenFeature([], tf.string),
                                                                    }
                                                                    ))

            dataset = dataset.map(lambda x:
                                  {
                                      'step_id': tf.cast(tf.reshape(x['step_id'], (1,)), tf.int32),
                                      'chunk_id': tf.cast(tf.reshape(x['chunk_id'], (1,)), tf.int32),
                                      'recording_id': tf.cast(tf.reshape(x['recording_id'], (1,)), tf.int32),
                                      'au': tf.reshape(tf.decode_raw(x['au'], tf.float32),
                                                       (-1, FEATURE_NUM["au"],)),
                                      'deepspectrum': tf.reshape(tf.decode_raw(x['deepspectrum'], tf.float32),
                                                                 (-1, FEATURE_NUM["deepspectrum"],)),
                                      'egemaps': tf.reshape(tf.decode_raw(x['egemaps'], tf.float32),
                                                            (-1, FEATURE_NUM["egemaps"],)),
                                      'fasttext': tf.reshape(tf.decode_raw(x['fasttext'], tf.float32),
                                                             (-1, FEATURE_NUM["fasttext"],)),
                                      'gaze': tf.reshape(tf.decode_raw(x['gaze'], tf.float32),
                                                         (-1, FEATURE_NUM["gaze"],)),
                                      'gocar': tf.reshape(tf.decode_raw(x['gocar'], tf.float32),
                                                          (-1, FEATURE_NUM["gocar"],)),
                                      'landmarks_2d': tf.reshape(tf.decode_raw(x['landmarks_2d'], tf.float32),
                                                                 (-1, FEATURE_NUM["landmarks_2d"],)),
                                      'landmarks_3d': tf.reshape(tf.decode_raw(x['landmarks_3d'], tf.float32),
                                                                 (-1, FEATURE_NUM["landmarks_3d"],)),
                                      'openpose': tf.reshape(tf.decode_raw(x['openpose'], tf.float32),
                                                             (-1, FEATURE_NUM["openpose"],)),
                                      'pdm': tf.reshape(tf.decode_raw(x['pdm'], tf.float32), (-1, FEATURE_NUM["pdm"],)),
                                      'pose': tf.reshape(tf.decode_raw(x['pose'], tf.float32),
                                                         (-1, FEATURE_NUM["pose"],)),
                                      'vggface': tf.reshape(tf.decode_raw(x['vggface'], tf.float32),
                                                            (-1, FEATURE_NUM["vggface"],)),
                                      'xception': tf.reshape(tf.decode_raw(x['xception'], tf.float32),
                                                             (-1, FEATURE_NUM["xception"],)),
                                      'support': tf.reshape(tf.decode_raw(x['support'], tf.float32), (-1, 1,)),
                                  }
                                  )
        elif task_name == "task3":
            dataset = dataset.map(lambda x: tf.parse_single_example(x,
                                                                    features=
                                                                    {
                                                                        'step_id': tf.FixedLenFeature([], tf.int64),
                                                                        'chunk_id': tf.FixedLenFeature([], tf.int64),
                                                                        'recording_id': tf.FixedLenFeature([],
                                                                                                           tf.int64),
                                                                        'au': tf.FixedLenFeature([], tf.string),
                                                                        'deepspectrum': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'egemaps': tf.FixedLenFeature([], tf.string),
                                                                        'fasttext': tf.FixedLenFeature([], tf.string),
                                                                        'gaze': tf.FixedLenFeature([], tf.string),
                                                                        'gocar': tf.FixedLenFeature([], tf.string),
                                                                        'landmarks_2d': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'landmarks_3d': tf.FixedLenFeature([],
                                                                                                           tf.string),
                                                                        'lld': tf.FixedLenFeature([], tf.string),
                                                                        'openpose': tf.FixedLenFeature([], tf.string),
                                                                        'pdm': tf.FixedLenFeature([], tf.string),
                                                                        'pose': tf.FixedLenFeature([], tf.string),
                                                                        'vggface': tf.FixedLenFeature([], tf.string),
                                                                        'xception': tf.FixedLenFeature([], tf.string),
                                                                        'support': tf.FixedLenFeature([], tf.string),
                                                                    }
                                                                    ))

            dataset = dataset.map(lambda x:
                                  {
                                      'step_id': tf.cast(tf.reshape(x['step_id'], (1,)), tf.int32),
                                      'chunk_id': tf.cast(tf.reshape(x['chunk_id'], (1,)), tf.int32),
                                      'recording_id': tf.cast(tf.reshape(x['recording_id'], (1,)), tf.int32),
                                      'au': tf.reshape(tf.decode_raw(x['au'], tf.float32), (FEATURE_NUM["au"],)),
                                      'deepspectrum': tf.reshape(tf.decode_raw(x['deepspectrum'], tf.float32),
                                                                 (FEATURE_NUM["deepspectrum"],)),
                                      'egemaps': tf.reshape(tf.decode_raw(x['egemaps'], tf.float32),
                                                            (FEATURE_NUM["egemaps"],)),
                                      'fasttext': tf.reshape(tf.decode_raw(x['fasttext'], tf.float32),
                                                             (FEATURE_NUM["fasttext"],)),
                                      'gaze': tf.reshape(tf.decode_raw(x['gaze'], tf.float32), (FEATURE_NUM["gaze"],)),
                                      'gocar': tf.reshape(tf.decode_raw(x['gocar'], tf.float32),
                                                          (FEATURE_NUM["gocar"],)),
                                      'landmarks_2d': tf.reshape(tf.decode_raw(x['landmarks_2d'], tf.float32),
                                                                 (FEATURE_NUM["landmarks_2d"],)),
                                      'landmarks_3d': tf.reshape(tf.decode_raw(x['landmarks_3d'], tf.float32),
                                                                 (FEATURE_NUM["landmarks_3d"],)),
                                      'lld': tf.reshape(tf.decode_raw(x['lld'], tf.float32), (FEATURE_NUM["lld"],)),
                                      'openpose': tf.reshape(tf.decode_raw(x['openpose'], tf.float32),
                                                             (FEATURE_NUM["openpose"],)),
                                      'pdm': tf.reshape(tf.decode_raw(x['pdm'], tf.float32), (FEATURE_NUM["pdm"],)),
                                      'pose': tf.reshape(tf.decode_raw(x['pose'], tf.float32), (FEATURE_NUM["pose"],)),
                                      'vggface': tf.reshape(tf.decode_raw(x['vggface'], tf.float32),
                                                            (FEATURE_NUM["vggface"],)),
                                      'xception': tf.reshape(tf.decode_raw(x['xception'], tf.float32),
                                                             (FEATURE_NUM["xception"],)),
                                      'support': tf.reshape(tf.decode_raw(x['support'], tf.float32), (1,)),
                                  }
                                  )
        else:
            raise NotImplementedError

    if task_name in ["task1", "task3"]:
        dataset = dataset.repeat()
        dataset = dataset.batch(seq_length)
        if is_training:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        padded_shapes = {
            'step_id': (None, 1),
            'chunk_id': (None, 1),
            'recording_id': (None, 1),
            'au': (None, None),
            'deepspectrum': (None, None),
            'egemaps': (None, None),
            'fasttext': (None, None),
            'gaze': (None, None),
            'gocar': (None, None),
            'landmarks_2d': (None, None),
            'landmarks_3d': (None, None),
            'lld': (None, None),
            'openpose': (None, None),
            'pdm': (None, None),
            'pose': (None, None),
            'vggface': (None, None),
            'xception': (None, None),
            'support': (None, 1),
        }

        if not ((split_name == "test") and (not are_test_labels_available)):
            padded_shapes["arousal"] = (None, 1)
            padded_shapes["valence"] = (None, 1)
            if task_name == "task3":
                padded_shapes["trustworthiness"] = (None, 1)

        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    elif task_name == "task2":
        dataset = dataset.repeat()
        if is_training:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        padded_shapes = {
            'step_id': (1,),
            'chunk_id': (1,),
            'recording_id': (1,),
            'au': (None, None),
            'deepspectrum': (None, None),
            'egemaps': (None, None),
            'fasttext': (None, None),
            'gaze': (None, None),
            'gocar': (None, None),
            'landmarks_2d': (None, None),
            'landmarks_3d': (None, None),
            'openpose': (None, None),
            'pdm': (None, None),
            'pose': (None, None),
            'vggface': (None, None),
            'xception': (None, None),
            'support': (None, 1),
        }
        if not ((split_name == "test") and (not are_test_labels_available)):
            padded_shapes["topic"] = (10, )
            padded_shapes["arousal"] = (3, )
            padded_shapes["valence"] = (3, )

        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    else:
        raise ValueError

    return dataset
