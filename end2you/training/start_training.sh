# Start training of the model. After each epoch evaluation is performed on the
# development set. The best found model is saved in the (`train_dir`/train/top_saved_models)

python train.py --dataset_dir='/path/to/tfrecords' \
                --train_dir='ckpt/' \
                --task='3'
