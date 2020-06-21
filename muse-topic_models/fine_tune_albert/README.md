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
4. Install Apex if you are using fp16 training. Please follow the instructions [here](https://github.com/NVIDIA/apex). 
5. Install all other packages (excluding Apex, installing Apex from pip has caused issues for several people.)
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
4. Install Apex if you are using fp16 training. Please follow the instructions [here](https://github.com/NVIDIA/apex). (Installing Apex from pip has caused issues for several people.)
5. Install simple transformers
`pip install simpletransformers` 
6. Install all other packages (excluding Apex, installing Apex from pip has caused issues for several people.)
`conda install --force-reinstall -y -q --name transformers -c conda-forge --file requirements.txt` 


## Training
Make sure data are in the dict specified in the processed_data_path argument (default = "../../data/processed_tasks/")
Start training of the baseline using the `run.py` file. For example:

```console
(transformers)$ python run.py --class_name arousal
```
Evaluation of the model after each epoch is performed using the development set. The best found model is saved in the end along with its performance.

The training script accepts the following list of arguments.

```
  -c, --class_name CLASS_NAME
                        specify which class of c2_muse_topic should be
                        predicted (arousal, valence, topic)
  -pd, --processed_data_path PROCESSED_DATA_PATH
                        specify the data folder
  -evaluate_test, --evaluate_test
                        specify if the model should evaluate on test (assuming labels are available).
  --model_type MODEL_TYPE
                        specify the transformer model
  --model_name MODEL_NAME
                        specify Transformer model name or path to Transformer model file.
```

## Score
Trained and the baseline models can be scored using predict.py. The data format is equivalent to the aggregated labels, thus, there must be a model for each label (arousal, valence, topic) in 'experiments/best_model/'. An aggregated file with the predictions is exported: 'experiments/predictions/args.predict_partition

### Output predictions
```console
(transformers)$ python predict.py --predict_partition test
```

### Output predictions and evaluate
```console
(transformers)$ python predict.py --predict_partition devel --evaluate
```

```
  -predict, --predict_partition PREDICT_PARTITION
                        specify the partition to be predicted.
  -evaluate, --evaluate
                        specify if the model should evaluate (assuming labels available).
  -pd PROCESSED_DATA_PATH, --processed_data_path PROCESSED_DATA_PATH
                        specify the data folder
  --model_type MODEL_TYPE
                        specify the transformer model
```

## Baseline models
The models can be downloaded here:

1. Download
2. Unzip in experiments/best_model/



