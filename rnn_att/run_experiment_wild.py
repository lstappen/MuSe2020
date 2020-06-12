from experiment_setup import train
from common import dict_to_struct

from configuration import *

#TODO: Models with masked test labels

def run(features_dict, task):
    results_lists = dict()
    results_lists["best_devel"] = dict()
    results_lists["test"] = dict()

    results_lists["best_devel"]["arousal"] = list()
    results_lists["best_devel"]["valence"] = list()
    results_lists["best_devel"]["score"] = list()

    results_lists["test"]["arousal"] = list()
    results_lists["test"]["valence"] = list()
    results_lists["test"]["score"] = list()

    for t in range(1):
        print("TRIAL:", t)
        # Make the arguments' dictionary.
        configuration = dict()
        configuration["challenge_folder"] = TASK_FOLDER["task" + repr(task)]
        configuration["tf_records_folder"] = TF_RECORDS_FOLDER["task" + repr(task)]
        configuration["output_folder"] = OUTPUT_FOLDER["task" + repr(task)]

        configuration["are_test_labels_available"] = False
        configuration["task"] = task
        configuration["use_data"] = features_dict
        configuration["model_type"] = "lstm"

        configuration["input_gaussian_noise"] = 0.1
        configuration["hidden_units"] = 40
        configuration["use_attention"] = True
        configuration["initial_learning_rate"] = 0.0001
        configuration["full_seq_length"] = SEQ_LEN
        configuration["train_batch_size"] = 50
        configuration["devel_batch_size"] = 50
        configuration["test_batch_size"] = 50
        configuration["num_epochs"] = 15
        configuration["patience"] = 10
        configuration["val_every_n_epoch"] = 1

        configuration["GPU"] = 0

        configuration = dict_to_struct(configuration)

        results = train(configuration)

        results_lists["best_devel"]["arousal"].append(results["arousal"]["best_devel_ccc"])
        results_lists["best_devel"]["valence"].append(results["valence"]["best_devel_ccc"])
        results_lists["best_devel"]["score"].append((results["arousal"]["best_devel_ccc"] + results["valence"]["best_devel_ccc"]) / 2.0)

        if configuration.are_test_labels_available:
            results_lists["test"]["arousal"].append(results["arousal"]["test_ccc"])
            results_lists["test"]["valence"].append(results["valence"]["test_ccc"])
            results_lists["test"]["score"].append(
                (results["arousal"]["test_ccc"] + results["valence"]["test_ccc"]) / 2.0)

    with open(results["method_string"] + ".txt", "w") as fp:
        for d_ccc_a, d_ccc_v, d_ccc in zip(results_lists["best_devel"]["arousal"],
                                           results_lists["best_devel"]["valence"],
                                           results_lists["best_devel"]["score"],
                                           ):
            fp.write(repr(d_ccc_a) + "\t" +
                     repr(d_ccc_v) + "\t" +
                     repr(d_ccc) + "\n")


def run_all():
    # TASK 1

    run({
        "au": True,
        "deepspectrum": True,
        "egemaps": False,
        "fasttext": True,
        "gaze": True,
        "gocar": True,
        "landmarks_2d": True,
        "landmarks_3d": True,
        "lld": False,
        "openpose": True,
        "pdm": True,
        "pose": True,
        "vggface": True,
        "xception": True
    }, task=1)


run_all()
