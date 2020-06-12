import tensorflow as tf
import numpy as np
import copy

from . import metrics
from .evaluation import Eval
from pathlib import Path

class EvalOnce(Eval):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @staticmethod
    def get_metric(metric):
        return {
            'ccc':metrics.np_concordance_cc2,
            'mse':metrics.np_mse,
            'uar':metrics.np_uar
        }[metric]
    
    @staticmethod
    def get_eval_tensors(sess, predictions, data_provider, evalute_path):
        
        dp_eval = copy.copy(data_provider)
        dp_eval.paths = [str(x) for x in Path(evalute_path).glob('*.tfrecords')]
        dp_eval.batch_size = 1
        dp_eval.is_training = False
        dp_eval.dataset_dir = evalute_path
        dp_eval.split_name = 'devel'
        dp_eval.num_examples = len(dp_eval.paths) # dp_eval.get_num_examples(evalute_path)
        
        frames, labels, sids = dp_eval.get_batch()
        
        get_pred = predictions(frames)
        
        seq_length = 1 if data_provider.seq_length == 0 \
            else data_provider.seq_length
        
        num_batches = int(np.ceil(
            dp_eval.num_examples / (dp_eval.batch_size * seq_length)))
        
        return get_pred, labels, sids, num_batches
    
    @staticmethod
    def eval_once(sess, get_pred, labels, sids, num_batches, num_outputs, metric_name):
        
        metric = EvalOnce.get_metric(metric_name)
        
        print('\n Start Evaluation \n')
        evaluated_predictions = []
        evaluated_labels = []
        for batch in range(num_batches):
            print('Example {}/{}'.format(batch+1, num_batches))
            preds, labs, s = sess.run([get_pred, labels, sids])
            evaluated_predictions.append(preds[0])
            evaluated_labels.append(labs[0])
        
        if 'uar' == metric_name:
            evaluated_predictions = np.vstack(evaluated_predictions)
            evaluated_labels = np.vstack(evaluated_labels)
            
            pred_i = {}
            lab_i = {}
            for i in range(2):
                start = i * 3
                end = (i+1) * 3
                
                pred_i[str(i)] = np.argmax(evaluated_predictions[:, start:end], -1).reshape((-1,1))
                lab_i[str(i)] = np.argmax(evaluated_labels[:, start:end], -1).reshape((-1,1))
            predictions = np.hstack([pred_i['0'], pred_i['1']])
            labels = np.hstack([lab_i['0'], lab_i['1']])
        else:
            predictions = np.vstack(evaluated_predictions).reshape((-1, num_outputs))
            labels = np.vstack(evaluated_labels).reshape((-1, num_outputs))
        
        mean_eval = 0
        for i in range(num_outputs):
            mean_eval += metric(predictions[:,i].reshape((-1,1)), labels[:,i].reshape((-1,1)))
        
        return mean_eval / float(num_outputs)

