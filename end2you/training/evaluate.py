import models
import data_provider
import argparse
import tensorflow as tf
import numpy as np


slim = tf.contrib.slim

parser = argparse.ArgumentParser(description='Evaluation flags.')
parser.add_argument('--checkpoint_path', type=str, default='ckpt/',
                    help='Path to checkpoint.')
parser.add_argument('--dataset_dir', type=str,
                    help='Directory with the tfrecords data.')
parser.add_argument('--task', type=str,
                    help='''Which task to train for. Takes values from [1-3]: 
                              1 - MuSe-Wild, \n
                              2 - MuSe-Topic, \n
                              3 - MuSe-Trust''')
parser.add_argument('--output_path', type=str, 
                    help='''Where to save the predictions of the model.
                            Saved as `taskI_predictions.csv` where I is the task.''')
parser.add_argument('--output', type=str, default='none',
                    help='''This flag is used only for the 3rd (MuSe-Trust) sub-challenge.
                            Output should be one of ['none', 'multitask'], where 'none'
                            indicates that only 'trustworthiness' will be predicted by the model, and
                            'multitask' indicates that the model will predict 'arousal', 'valence'
                            and 'trustworthiness' (default 'none').''')

def start_evaluation(args):
    if args.task == '1':
        out_names = ['arousal', 'valence'] 
    elif args.task == '2':
        out_names = ['arousal', 'valence']      
    elif args.task == '3':
        out_names = ['trustworthiness']
    else:
        raise ValueError('Task must be in the range [1-3]')
    
    if args.task == '3':
        provider = data_provider.get_provider(
            '3', args.output, args.dataset_dir, 'test', 1, False)
    else:
        provider = data_provider.get_provider(
            args.task, args.dataset_dir, 'test', 1, False)
    num_batches = provider.get_num_examples()
    
    model_inp, _, _ = provider.get_batch()
    
    # Define model graph.
    with slim.arg_scope([slim.layers.batch_norm, slim.layers.dropout],
        is_training=False):
          prediction = models.get_model(args.task, provider.num_outs, model_inp)
    
    coord = tf.train.Coordinator()
    
    saver = tf.train.Saver()
    print('Loading model from {}'.format(args.checkpoint_path))
    
    with tf.Session() as sess:
        try:
            saver.restore(sess, args.checkpoint_path)
            tf.train.start_queue_runners(sess=sess)
            evaluated_preds = []
            evaluated_gts = []
            
            for k in range(num_batches):
                pr = sess.run(prediction)
                evaluated_preds.append(pr[0])
            coord.request_stop()
        except Exception as e:
            print('Exception ', e)
            coord.request_stop()
    coord.request_stop()
    predictions = np.vstack(evaluated_preds)
    
    if args.task == '3' and args.output == 'multitask':
        predictions = predictions[:, 0]

    np.savetxt('task{}_predictions.csv'.format(args.task), 
        predictions, delimiter=',')

if __name__ == '__main__':
    args = parser.parse_args()
    start_evaluation(args)
