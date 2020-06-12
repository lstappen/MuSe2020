# Code to replicate the RNN+ATT baseline experiments of MuSe 2020

## Installation setup
Experiments were performed using Python 3.5.2 and tensorflow-gpu 1.15.2

## Run baselines
- Add the challenge data in a folder: hence CHALLENGE_FOLDER
- Setup the directory location in configuration.py
- Run data_generator.py to extract tf records files (and wait awhile...)
- Config & Run experiments for Tasks 1 (wild), 2 (topic), 3 (trust) by running the run_experiment_taskX.py scripts

## Contact
georgios.rizos12@imperial.ac.uk