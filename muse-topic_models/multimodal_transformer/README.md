# MuSe Challenge - MulT model

The Multimodal Transformer for Unaligned Multimodal Language Sequences (https://github.com/yaohungt/Multimodal-Transformer/, https://arxiv.org/pdf/1906.00295.pdf) serves as a base for evaluating unaligned fusion of multi-modalities for the classification task (MuSe-Topic) of the of the [MuSe](https://www.muse-challenge.org/) challenge. 

## Installation

Check out the original repo for installation instructions. 

## Prepare data
Make sure data are in the dict specified in the processed_data_path argument (default = "../../data/processed_tasks/")

Start generating the serialised data of the baseline using the `prepare.py` file. For example:
```console
(mml)$ python prepare.py --class_name arousal
```
Specify the feature !path! to create specific baseline data:

```
  -c, --class_name CLASS_NAME
                        specify which class of task 2 should be predicted
  -a, --aligned         specify if data are aligned
  -at, --alignment_type ALIGNMENT_TYPE
                        specify the alignment
  -cf, --data_dir DATA_DIR
                        specify the task folder
  -vi, --vision_path VISION_PATH
                        specify the vision folder
  -te, --text_path TEXT_PATH
                        specify the text folder
  -au, --audio_path AUDIO_PATH
                        specify the audio folder
  -out, --output_path OUTPUT_PATH
                        specify the output directory
  -exa, --example_mode  specify if test mode is activated (only few test data)
  -et, --evaluate_test  specify if the model should run and evaluate on test
                        (assuming labels available).

```

Before create the file over the whole data set you can run a test using --example_mode.

## Training data

Start training with the baseline using the `main.py` file. For example:

```console
(mml)$ python main.py --experiment_name std --class_name arousal
```
Evaluation of the model after each epoch is performed using the development set. The best found model is saved in the end along with its performance.

The training script accepts the following list of arguments.

```
  -c, --class_name CLASS_NAME
                        specify which class of task 3 should be predicted
  -n, --experiment_name EXPERIMENT_NAME
                        specify the name of the experiment
  -evaluate_test, --evaluate_test
                        specify if the model should evaluate on test (assuming
                        labels available).
  -a, --aligned         specify if data is aligned
  -vi, --vision VISION  specify the feature name
  -te, --text TEXT      specify the feature name
  -au, --audio AUDIO    specify the feature name
  -ca, --cache          specify if data preprocessing should be stored

```
We recommend the --cache parameter if you run multiple experiments with the same feature sets combination.


## Score
Trained and the baseline models can be scored using predict.py. The data format is equivalent to the aggregated labels, thus, there must be a model for each label (arousal, valence, topic) in 'experiments/pretrained_model/'. An aggregated file with the predictions is exported: 'experiments/predictions/args.predict_partition. Be aware that some baseline experiment_names have different names.

### Output predictions
```console
(mml)$ python predict.py --experiment_name std --predict_partition test
```

### Output predictions and evaluate
```console
(mml)$ python predict.py --experiment_name std --predict_partition devel --evaluate
```

```
 -n, --experiment_name EXPERIMENT_NAME
                        specify the name of the experiment
  -pd, --processed_data_path PROCESSED_DATA_PATH
                        specify the data folder
  -a, --aligned         specify if data is aligned
  -vi, --vision VISION  specify the feature name
  -te, --text TEXT      specify the feature name
  -au, --audio AUDIO    specify the feature name
  -predict, --predict_partition PREDICT_PARTITION
                        specify the partition to be predicted.
  -evaluate, --evaluate
                        specify if the model should evaluate (assuming labels
                        available).

```

## Baseline models
The models can be downloaded here:

1. Download https://megastore.uni-augsburg.de/get/0ucLK3P_7a/
2. Unzip in experiments/pretrained_model/

