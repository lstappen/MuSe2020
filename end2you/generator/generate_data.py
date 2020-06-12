import argparse
import data_generator as dg

from pathlib import Path


parser = argparse.ArgumentParser(description='Data generator flags.')
parser.add_argument('--raw_data_path', type=str,
                    help='Path to directory with raw data.')
parser.add_argument('--tfrecord_folder', type=str,
                    help='Path to save tfrecords.')
parser.add_argument('--task', type=str,
                    help='''Which task to train for. Takes values from [1-3]: 
                              1 - MuSe-Wild, \n
                              2 - MuSe-Topic, \n
                              3 - MuSe-Trust''')

def get_generator_params(args):
    generator_params = {}
    generator_params['task'] = args.task
    generator_params['tfrecord_folder'] = args.tfrecord_folder
    generator_params['root_dir'] = args.raw_data_path
    
    return generator_params


if __name__ == '__main__':
    args = parser.parse_args()
    get_generator_params = get_generator_params(args)
    generator = dg.DataGenerator(**get_generator_params)
    generator.create_tfrecords()
