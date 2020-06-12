# Perform evaluation of a saved model.

python evaluate.py --checkpoint_path='/path/to/saved/model/model.ckpt-XXXX' \
                   --dataset_dir='/path/to/tfrecords/folder' \
                   --task='1' \
                   --output_path='./'
