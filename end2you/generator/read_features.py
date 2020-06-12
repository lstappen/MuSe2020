import numpy as np

from pathlib import Path


class ReadFeatures:
    
    def __init__(self, root_dir, task):
        self.label_path = root_dir / 'label_segments'
        self.feats_path = root_dir / 'feature_segments' / 'label_aligned'
        self.task = task
        self.root_dir = root_dir
        
        if task == '1' or task == '2':
            self.features = ['arousal', 'valence']
        elif task =='3':
            self.features = ['arousal', 'valence', 'trustworthiness']
        
        if task == '2':
            self._get_visual_feats = self._get_feats
            self._get_text_feats = self._get_feats
            self._get_av_dict = self._get_av_dict_t2
        else:
            self._get_visual_feats = self._get_feats
            self._get_text_feats = self._get_feats
            self._get_av_dict = self._get_av_dict_t13
    
    def _get_feats(self, folder, mod):

        file_path = self.feats_path / mod / '{}.csv'.format(folder)
        dim_array = np.loadtxt(str(file_path), skiprows=1, delimiter=',', dtype=str)
        dim_dict = {x:[] for x in np.unique(dim_array[:, 1])}
        
        for i, x in enumerate(dim_array[:, 1]):
            dim_dict[x].append(np.float32(dim_array[i, 2:]))
        
        timestamps = {}
        for x in np.unique(dim_array[:, 1]):
            idx = np.where(dim_array[:,1] == x)
            timestamps[x] = dim_array[idx,0].astype(np.int32)
        
        return dim_dict, timestamps
    
    
    def _get_av_dict_t2(self, file_path):
        
        dim_array = np.loadtxt(str(file_path), skiprows=1, delimiter=',', dtype=str)
        dim_dict = {dim_array[i,2]:dim_array[i,3] for i in range(dim_array.shape[0])}

        return dim_dict
    
    def _get_av_dict_t13(self, file_path):
        
        dim_array = np.loadtxt(str(file_path), skiprows=1, delimiter=',', dtype=str)
        dim_dict = {x:[] for x in np.unique(dim_array[:, 2])}
        for timestamp, value, x in dim_array:
            dim_dict[x].append((np.int32(timestamp), np.float32(value)))
        
        return dim_dict
    
    def _get_visual_feats_t13(self, folder, mod=None):
        
        visual_feats_file = self.feats_path / 'vggface' / (folder + '.csv') 
        videos_visual_feats = np.loadtxt(str(visual_feats_file), skiprows=1, delimiter=',', dtype=str)
        
        arousal_file = self.label_path / 'arousal' / '{}.csv'.format(folder)
        arousal_csv = np.loadtxt(str(arousal_file), skiprows=1, delimiter=',', dtype=str)
        visual_feats = {} 
        timestamps = {}
        for arousal_segid in np.unique(arousal_csv[:,2]):
            visual_feats[arousal_segid] = []
            segid_idx = np.where(videos_visual_feats[:,1] == arousal_segid)
            visual_feats_values = videos_visual_feats[segid_idx]
            arousal_segid_idx = np.where(arousal_csv[:,2] == arousal_segid)
            j = 0
            timestamps[arousal_segid] = arousal_csv[arousal_segid_idx,0][0]
            for arousal_timestamp in arousal_csv[arousal_segid_idx,0][0]:
                
                timestamp = videos_visual_feats[j,0]
                
                if arousal_timestamp != timestamp:
                    feats = np.zeros((1, 512), dtype=np.float32)
                else:
                    feats = visual_feats_values[j, 2:].astype(np.float32).reshape((1,-1))
                    j += 1
                visual_feats[arousal_segid].append(feats)
        
        return visual_feats, timestamps
    
    def _get_text_feats_t13(self, folder, mod=None):
        
        text_feats_file = self.feats_path / 'fasttext' / (folder + '.csv') 
        all_text_feats = np.loadtxt(str(text_feats_file), skiprows=1, delimiter=',', dtype=str)
        
        arousal_file = self.label_path / 'arousal' / '{}.csv'.format(folder)
        arousal_csv = np.loadtxt(str(arousal_file), skiprows=1, delimiter=',', dtype=str)
        text_feats = {} 
        for arousal_segid in np.unique(arousal_csv[:,2]):
            text_feats[arousal_segid] = []
            segid_idx = np.where(all_text_feats[:,2] == arousal_segid)
            text_feats_values = all_text_feats[segid_idx]
            arousal_segid_idx = np.where(arousal_csv[:,2] == arousal_segid)
            if text_feats_values.shape[0] == 0:
                text_feats[arousal_segid].append(np.zeros_like((len(arousal_segid_idx), 300)))
                continue
            
            j = 0
            for arousal_timestamp in arousal_csv[arousal_segid_idx,0][0]:
                timestamp = int(text_feats_values[j,1])
                if int(arousal_timestamp) < timestamp:
                    feats = text_feats_values[j, 3:].astype(np.float32).reshape((1,-1))
                else:
                    j = (j + 1) if text_feats_values.shape[0] > (j+1) else j
                    feats = text_feats_values[j, 3:].astype(np.float32).reshape((1,-1))
                text_feats[arousal_segid].append(feats)
        
        return text_feats
    
