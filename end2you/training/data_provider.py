import tensorflow as tf
import numpy as np

from pathlib import Path


def get_provider(task, *args, **kwargs):
    return {
        '1':Task1Provider,
        '2':Task2Provider,
        '3':Task3Provider,
    }[task](*args, **kwargs)


class DataProvider:
    
    def __init__(self,
                 dataset_dir,
                 split_name,
                 batch_size,
                 is_training):
        
        self.batch_size = batch_size
        self.is_training = is_training
        self.seq_length = 0
        
        self.root_path = Path(dataset_dir) / split_name
        
        self.paths = [str(y) for y in self.root_path.glob('*.tfrecords')]
        
        self.num_examples = len(self.paths)
    
    def get_num_examples(self, eval_path=None):
        return self.num_examples


class Task1Provider(DataProvider):
    
    def __init__(self, *args, **kwargs):
        self.label_shape = 2
        self.num_outs = 2
        super().__init__(*args, **kwargs)
    
    def get_batch(self):
        filename_queue = tf.train.string_input_producer(self.paths, shuffle=self.is_training)
        
        reader = tf.TFRecordReader()
        
        _, serialized_example = reader.read(filename_queue)
        
        features = tf.parse_single_example(
            serialized_example,
            features={
                'raw_waveform': tf.FixedLenFeature([], tf.string),
                'visual_features': tf.FixedLenFeature([], tf.string),
                'text_features': tf.FixedLenFeature([], tf.string),
                'arousal': tf.FixedLenFeature([], tf.string),
                'valence': tf.FixedLenFeature([], tf.string)
            }
        )
        
        raw_waveform = tf.decode_raw(features['raw_waveform'], tf.float32)
        visual_features = tf.decode_raw(features['visual_features'], tf.float32)
        text_features = tf.decode_raw(features['text_features'], tf.float32)
        arousal = tf.decode_raw(features['arousal'], tf.float32)
        valence = tf.decode_raw(features['valence'], tf.float32)
        seq_len = tf.shape(valence)[0]
        
        raw_waveform, visual_features, text_features, arousal, valence, seq_len = tf.train.batch(
            [raw_waveform, visual_features, text_features, arousal, valence, seq_len], 
            self.batch_size, dynamic_pad=True)
        
        raw_waveform = tf.reshape(raw_waveform, (self.batch_size, -1, 2000))
        visual_features = tf.reshape(visual_features, (self.batch_size, -1, 512))
        text_features = tf.reshape(text_features, (self.batch_size, -1, 300))
        
        arousal = tf.reshape(arousal, (self.batch_size, -1, 1))
        valence = tf.reshape(valence, (self.batch_size, -1, 1))
        ground_truth = tf.concat([arousal, valence], axis=2)
        seq_len = tf.reshape(seq_len, (self.batch_size, -1, 1))
        
        return [raw_waveform, visual_features, text_features], ground_truth, seq_len


class Task2Provider(DataProvider):
    
    def __init__(self, *args, **kwargs):
        self.num_classes = 3
        self.num_outs = 6
        super().__init__(*args, **kwargs)
    
    def get_batch(self):
        filename_queue = tf.train.string_input_producer(self.paths, shuffle=self.is_training)
        
        reader = tf.TFRecordReader()
        
        _, serialized_example = reader.read(filename_queue)
        
        features = tf.parse_single_example(
            serialized_example,
            features={
                'raw_waveform': tf.FixedLenFeature([], tf.string),
                'visual_features': tf.FixedLenFeature([], tf.string),
                'text_features': tf.FixedLenFeature([], tf.string),
                'arousal': tf.FixedLenFeature([], tf.string),
                'valence': tf.FixedLenFeature([], tf.string)
            }
        )
        
        raw_waveform = tf.decode_raw(features['raw_waveform'], tf.float32)
        visual_features = tf.decode_raw(features['visual_features'], tf.float32)
        text_features = tf.decode_raw(features['text_features'], tf.float32)
        arousal = tf.one_hot(tf.cast(
                 tf.decode_raw(features['arousal'], tf.float32), tf.int32), 3)
        valence = tf.one_hot(tf.cast(
                 tf.decode_raw(features['valence'], tf.float32), tf.int32), 3)
        seq_len = tf.shape(valence)[0]
        
        raw_waveform, visual_features, text_features, arousal, valence, seq_len = tf.train.batch(
            [raw_waveform, visual_features, text_features, arousal, valence, seq_len], 
            self.batch_size, dynamic_pad=True)
        
        raw_waveform = tf.reshape(raw_waveform, (self.batch_size, -1, 2000))
        
        visual_features = tf.reshape(visual_features, (self.batch_size, -1, 512))
        text_features = tf.reshape(text_features, (self.batch_size, -1, 300))
        
        ground_truth = tf.concat([arousal, valence], axis=2)
        seq_len = tf.reshape(seq_len, (self.batch_size, -1, 1))
        
        return [raw_waveform, visual_features, text_features], ground_truth, seq_len


class Task3Provider(DataProvider):
    
    def __init__(self, gt, *args, **kwargs):
        self.label_shape = 1 if gt == 'none' else 3
        
        if gt not in ['multitask', 'none']:
            raise ValueError('Argument gt should be one of: [multitask, none]')
        self.gt = gt
        self.num_outs = 1 if gt == 'none' else 3

        super().__init__(*args, **kwargs)
    
    def get_batch(self):
        filename_queue = tf.train.string_input_producer(self.paths, shuffle=self.is_training)
        
        reader = tf.TFRecordReader()
        
        _, serialized_example = reader.read(filename_queue)
        
        features = tf.parse_single_example(
            serialized_example,
            features={
                'raw_waveform': tf.FixedLenFeature([], tf.string),
                'visual_features': tf.FixedLenFeature([], tf.string),
                'text_features': tf.FixedLenFeature([], tf.string),
                'arousal': tf.FixedLenFeature([], tf.string),
                'valence': tf.FixedLenFeature([], tf.string),
                'trustworthiness': tf.FixedLenFeature([], tf.string)
            }
        )
        
        raw_waveform = tf.decode_raw(features['raw_waveform'], tf.float32)
        visual_features = tf.decode_raw(features['visual_features'], tf.float32)
        text_features = tf.decode_raw(features['text_features'], tf.float32)
        arousal = tf.decode_raw(features['arousal'], tf.float32)
        valence = tf.decode_raw(features['valence'], tf.float32)
        trustworthiness = tf.decode_raw(features['trustworthiness'], tf.float32)
        seq_len = tf.shape(valence)[0]
        
        raw_waveform, visual_features, text_features, arousal, valence, seq_len, trustworthiness = tf.train.batch(
            [raw_waveform, visual_features, text_features, arousal, valence, seq_len, trustworthiness], 
            self.batch_size, dynamic_pad=True)
        
        raw_waveform = tf.reshape(raw_waveform, (self.batch_size, -1, 2000))
        
        visual_features = tf.reshape(visual_features, (self.batch_size, -1, 512))
        text_features = tf.reshape(text_features, (self.batch_size, -1, 300))
        
        arousal = tf.reshape(arousal, (self.batch_size, -1, 1))
        valence = tf.reshape(valence, (self.batch_size, -1, 1))

        ground_truth = tf.reshape(trustworthiness, (self.batch_size, -1, 1))
        if self.gt == 'multitask':
            ground_truth = tf.concat([ground_truth, arousal, valence], axis=2)

        seq_len = tf.reshape(seq_len, (self.batch_size, -1, 1))
        
        return [raw_waveform, visual_features, text_features], ground_truth, seq_len
