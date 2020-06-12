# MuSe Challenge - End2You Baseline

We introduce the [End2You](https://github.com/end2you/end2you) baseline for the [MuSe](https://www.muse-challenge.org/) challenge. 
End2You is an open-source toolkit for multimodal profiling by end-to-end deep learning. 

For all tasks we utilize three modalities, namely, audio, visual and textual. Our audio model is inspired by recently proposed emotion recognition model~\cite{tzirakis2018end}, and is comprised of a convolution recurrent neural network. In particular, we use 3 convolution layers to extract spatial features from the raw segments. Our visual information is comprised of the VGG face features, where we use zero vectors when the face was not detected in a frame. Finally, as text features we use fasttext, where we replicate the text features that span several segments. 
We concatenate all uni-modal features and feed them to a 1 layer LSTM to capture the temporal dynamics in the data before the final prediction. 


## Trained Models
Trained models of the baselines for all sub-challenges can be found here:

http://www.doc.ic.ac.uk/~pt511/MuSe/End2You_models/

## Installation
We highly recommended to use [conda](http://conda.pydata.org/miniconda.html) as your Python distribution.
Once downloading and installing [conda](http://conda.pydata.org/miniconda.html), this project can be installed by:

**Step 1:** Create a new conda environment and activate it:
```console
$ conda create -n muse_e2u python=3.5
$ source activate muse_e2u
```

**Step 2:** Install [TensorFlow v.1.15](https://www.tensorflow.org/). 
For example, for Linux, the installation of GPU enabled, Python 3.5 TensorFlow involves:
```console
(muse_e2u)$ pip install tensorflow-gpu==1.15
```

## Generating Data
Create tfrecords files using the `start_generator.sh` file in the `generator` forlder. For example:
```console
(muse_e2u)$ python generate_data.py --raw_data_path='/path/to/raw_data/' \
                                    --tfrecord_folder='/path/to/save/tfrecords' \
                                    --task='1'
```

The `generate_data.py` script accepts the following list of arguments.

```
  --raw_data_path RAW_DATA_PATH
                        Path to directory with raw data.
  --tfrecord_folder TFRECORD_FOLDER
                        Path to save tfrecords.
  --task TASK           Which task to train for. Takes values from [1-3]: 
                         1 - MuSe-Wild, \n
                         2 - MuSe-Topic, \n
                         3 - MuSe-Trust
```

## Training
Start training with the baseline using the `start_training.sh` file in the `training` forlder. For example:
```console
(muse_e2u)$ python train.py --dataset_dir='/path/to/tfrecords' \
                            --train_dir='ckpt/' \
                            --task='3' 
```
Evaluation of the model after each epoch is performed using the development set. The best found model is saved in the end along with its performance.

The `train.py` script accepts the following list of arguments.

```
  --initial_learning_rate INITIAL_LEARNING_RATE
                        Initial learning rate.
  --batch_size BATCH_SIZE
                        The batch size to use.
  --train_dir TRAIN_DIR
                        Directory where to write event logs and checkpoint.
  --dataset_dir DATASET_DIR
                        The tfrecords directory.
  --task TASK           Which task to train for. Takes values from [1-3]: 
                         1 - MuSe-Wild, \n
                         2 - MuSe-Topic, \n
                         3 - MuSe-Trust  --num_epochs NUM_EPOCHS
                        The number of epochs to train model.
   --output OUTPUT
                        This flag is used only for the 3rd (MuSe-Trust) sub-challenge.
                        Output should be one of ['none', 'multitask], where 'none'
                        indicates that only 'trustworthiness' will be predicted by the model, and
                        'multitask' indicates that the model will predict 'arousal', 'valence'
                        and 'trustworthiness' (default 'none').
```

## Evaluation
Start training with the baseline using the `start_training.sh` file in the `training` forlder. For example:
```console
(muse_e2u)$ python evaluate.py --dataset_dir='/path/to/tfrecords' \
                               --train_dir='ckpt/' \
                               --task='3' 
```

The `evaluate.py` script accepts the following list of arguments.


```
  --checkpoint_path CHECKPOINT_PATH
                        The path to the saved model.
  --dataset_dir DATASET_DIR
                        The tfrecords directory.
  --task TASK           Which task to train for. Takes values from [1-3]: 
                         1 - MuSe-Wild, \n
                         2 - MuSe-Topic, \n
                         3 - MuSe-Trust
  --output_path OUTPUT_PATH
                        The path to save the predictions of the model.
                        Model is saved to 'output_path/taskI_predictions.csv'
                        where I is the task number.
   --output OUTPUT
                        This flag is used only for the 3rd (MuSe-Trust) sub-challenge.
                        Output should be one of ['none', 'multitask], where 'none'
                        indicates that only 'trustworthiness' will be predicted by the model, and
                        'multitask' indicates that the model will predict 'arousal', 'valence'
                        and 'trustworthiness' (default 'none').
```
