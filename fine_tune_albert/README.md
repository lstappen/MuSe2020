# MuSe Challenge - fine-tuned Albert Baseline

We introduce the fine-tuned Albert baseline for the MuSe-Topic task of the [MuSe](https://www.muse-challenge.org/) challenge. 
It is build up on open-source high-level APIs Simple Transformers (https://github.com/ThilinaRajapakse/simpletransformers) and HuggingFace (https://github.com/huggingface/transformers). For all tasks we utilize the text modality only.


## Installation

### virtenv guide
1. Make sure your GPU (cuda), pytorch, tensorflow are set up. 
2. Create a virtenv 
`python3 -m venv /path/to/new/virtual/environment/albert`
3. Activate virtenv
`source ./path/to/new/virtual/environment/albert/bin/active`
3. Install Apex if you are using fp16 training. Please follow the instructions [here](https://github.com/NVIDIA/apex). 
4. Install all other packages (excluding Apex, installing Apex from pip has caused issues for several people.)
`pip3 install -r requirements.txt`

### virtenv conda
1. Make sure your GPU (cuda), pytorch, tensorflow are set up. 
2. Install Anaconda or Miniconda Package Manager from [here](https://www.anaconda.com/distribution/)
3. Create a new virtual environment and install packages.  
`conda create -n transformers python pandas tqdm`  
`conda activate transformers`  
If using cuda:  
&nbsp;&nbsp;&nbsp;&nbsp;`conda install pytorch cudatoolkit=10.1 -c pytorch`  
else:  
&nbsp;&nbsp;&nbsp;&nbsp;`conda install pytorch cpuonly -c pytorch`  
3. Install Apex if you are using fp16 training. Please follow the instructions [here](https://github.com/NVIDIA/apex). (Installing Apex from pip has caused issues for several people.)
4. Install simple transformers
`pip install simpletransformers` 
5. Install all other packages (excluding Apex, installing Apex from pip has caused issues for several people.)
`conda install --force-reinstall -y -q --name transformers -c conda-forge --file requirements.txt` 


## Training
Make sure data are in the dict specified in the processed_data_path argument (default = "../../data/processed_tasks/")
Start training with the baseline using the `run.py` file. For example:

```console
(transformers)$ python run.py --class_name arousal
```
Evaluation of the model after each epoch is performed using the development set. The best found model is saved in the end along with its performance.

The training script accepts the following list of arguments.

```
  --initial_learning_rate INITIAL_LEARNING_RATE
                        Initial learning rate.
  --batch_size BATCH_SIZE
                        The batch size to use.
  --train_dir TRAIN_DIR
                        Directory where to write event logs and checkpoint.
  --dataset_dir DATASET_DIR
                        The tfrecords directory.
  --task TASK           The task to execute. `1`, `2`, or `3`.
  --num_epochs NUM_EPOCHS
                        The number of epochs to train model.
```

## Score
Trained and the baseline models can be score using predict.py. The data format is equivalent to the aggregated labels, thus, there must be a model for each label (arousal, valence, topic) in 'experiments/best_model/'. An aggregated file with the predictions is exported here: 'experiments/predictions/args.predict_partition

### Output predictions
```console
(transformers)$ python predict.py --predict_partition test
```

### Output predictions and evaluate
```console
(transformers)$ python predict.py --predict_partition devel --evaluate
```

## Baseline models
The models can be downloaded here:

1. Download
2. Unzip in experiments/best_model/



