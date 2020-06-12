import numpy as np
import librosa as lb
import tensorflow as tf
import os

from pathlib import Path
from read_features import ReadFeatures


class DataGenerator:
    
    def __init__(self,
                 root_dir,
                 tfrecord_folder,
                 task,
                 SR=8000):
        """ Superclass that contains main functionality for task specific data generator classes.
         Args:
           root_dir: Path to task folder.
           tfrecord_folder: Path to save tfrecord folder.
           task: Task to generate data from.
           SR: Sampling rate (default 8kHz).
        """

        self.task = task
        root_dir = Path(root_dir)
        org_task = {'1':'c1_muse_wild', '2':'c2_muse_topic', '3':'c3_muse_trust'}[task]

        self.tfrecord_split_folder = Path(tfrecord_folder)
        self.tfrecord_split_folder.mkdir(exist_ok=True)
        
        (self.tfrecord_split_folder / 'train').mkdir(exist_ok=True)
        (self.tfrecord_split_folder / 'devel').mkdir(exist_ok=True)
        (self.tfrecord_split_folder / 'test').mkdir(exist_ok=True)
        
        self.feats_extract = ReadFeatures(root_dir / '{}'.format(org_task), task)
        self.audio_segments = list(Path(root_dir / org_task / 'audio_segments').glob('*'))

        if len(self.audio_segments) == 0:
            raise Exception('Empty audio segments folder!')
        
        self.SR = SR
        
        partitions_csv = np.loadtxt(str(Path(root_dir) /  'metadata/partition.csv'), skiprows=1, delimiter=',', dtype=str)
        
        self.split = {}
        for folder, partition in partitions_csv:
            self.split[folder] = partition
    
    def _bytes_feauture(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _serialize_sample(self, tf_file, raw_waveform, visual_features, text_feats, kwargs):
        writer = tf.python_io.TFRecordWriter((str(tf_file) + '.tfrecords'))
        values = {
            name: self._bytes_feauture(kwargs[name].tobytes()) for name in kwargs.keys()
        }
        
        feature_dict = {
                'raw_waveform': self._bytes_feauture(raw_waveform.tobytes()),
                'visual_features': self._bytes_feauture(visual_features.tobytes()),
                'text_features': self._bytes_feauture(text_feats.tobytes()),
        }
        feature_dict.update(values)
        
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        
        writer.write(example.SerializeToString())
    
    def del_long_segs(self, raw_shapes):

        del_files = []
        for shape in raw_shapes:
            if shape[2][0] > 400 or shape[2][0] <= 2:
                try:
                    file = str(shape[1] / (shape[0].name[:-4] + '.tfrecords'))
                    os.system('rm {}'.format(file))
                    del_files.append(file)
                except Exception as e:
                    print('Exception!', e)
                    continue
    
    def create_tfrecords(self):
        
        faulty_segs = []
        raw_shapes = []
        for audio_folder in self.audio_segments:
            folder = audio_folder.name
            print('Creating tfrecords for folder {}'.format(folder))
            
            try:
                visual_feats, timestamps = self.feats_extract._get_visual_feats(folder, 'vggface')
                text_feats, _ = self.feats_extract._get_text_feats(folder, 'fasttext')
                
                if self.split[folder] == 'test':
                    values = {name:{} for name in self.feats_extract.features}
                    for name in self.feats_extract.features:
                        for seg_id in visual_feats.keys():
                            val = np.zeros((len(visual_feats[seg_id]), 1)).astype(np.float32)
                            t = np.array(timestamps[seg_id]).reshape((-1,1))
                            values[name][seg_id] = np.hstack([t, val])
                else:
                    values = {
                        name:self.feats_extract._get_av_dict(
                            self.feats_extract.label_path / name / '{}.csv'.format(folder)) for name in self.feats_extract.features
                    }
            except Exception as e:
                print('0', e)
                continue
            
            segment_len = 0.25 * self.SR
            for audio_file in audio_folder.glob('*'):
                tf_file = self.tfrecord_split_folder / self.split[folder]
                if (tf_file / '{}.tfrecords'.format(audio_file.name[:-4])).exists():
                    continue
                
                # Raw waveform
                raw_waveform, sr = lb.core.load(str(audio_file), sr=self.SR)
                segment_id = audio_file.name[:-4].split('_')[1]
                
                # array with cols: timestamp, value
                try:
                    wav_values = {
                        name:np.array(values[name][segment_id]).astype(np.float32) for name in self.feats_extract.features
                    }
                except Exception as e:
                    faulty_segs.append((segment_id, audio_folder, audio_file))
                    continue
                
                if self.task != '2':
                    duration = (timestamps[segment_id][0,-1] - timestamps[segment_id][0,0])
                    num_segments = int((duration/1000*sr) / segment_len) + 1
                    segments = np.linspace(0, duration/1000*sr, num=num_segments, dtype=int)
                else:
                    num_segments = int((raw_waveform.shape[0]) / segment_len)
                    segments = np.linspace(0, segment_len * num_segments, num=num_segments, dtype=int, endpoint=False)
                
                raw_segments = []
                for idx in range(num_segments - 1):
                    start = segments[idx]
                    stop = segments[idx+1]
                    
                    raw_segments.append(raw_waveform[start:stop])
                
                segs_values = {}
                for name in self.feats_extract.features:
                    if self.task == '2':
                        segs_values[name] = np.array(wav_values[name]).reshape((-1,1)).astype(np.float32) 
                    else:
                        segs_values[name] = np.array(wav_values[name])[1:,1].reshape((-1,1)).astype(np.float32) 
                        assert segs_values[name].shape[0] == len(raw_segments)

                try:
                    seg_visual_feats = np.vstack(visual_feats[segment_id])[1:].astype(np.float32)
                    seg_text_feats = np.vstack(text_feats[segment_id])[1:].astype(np.float32)
                except Exception as e:
                    faulty_segs.append((audio_folder, audio_file))
                    print('2', e)
                    continue
                
                try:
                    raw_segments = np.array(raw_segments).astype(np.float32)
                    raw_shape = raw_segments.shape
                    if raw_segments.shape[0] != seg_visual_feats.shape[0] or raw_segments.shape[0] != seg_text_feats.shape[0]:
                        continue
                    raw_shapes.append([audio_file, tf_file, raw_shape])
                except Exception as e:
                    faulty_segs.append((audio_folder, audio_file))
                    print('3',e)
                    continue
                
                if seg_visual_feats.shape[0] == raw_segments.shape[0] + 1:
                    seg_visual_feats = seg_visual_feats[:-1]
                    seg_text_feats = seg_text_feats[:-1]
                
                if seg_text_feats.shape[0] == 0:
                    seg_text_feats = np.zeros((seg_visual_feats.shape[0], 300)).astype(np.float32)
                
                # Create tfrecord
                tf_file.mkdir(exist_ok=True)
                self._serialize_sample(tf_file / audio_file.name[:-4],
                    raw_segments, seg_visual_feats, seg_text_feats, segs_values)
        
        self.del_long_segs(raw_shapes)

