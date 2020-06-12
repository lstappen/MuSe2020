from experiment_setup import train
from common import dict_to_struct

from configuration import *
#TODO: Models with masked test labels

def run(features_dict, task):
    results_lists = dict()
    results_lists["best_devel"] = dict()
    results_lists["test"] = dict()

    results_lists["best_devel"]["trustworthiness"] = list()
    results_lists["test"]["trustworthiness"] = list()

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

        configuration["GPU"] = 1

        configuration = dict_to_struct(configuration)

        results = train(configuration)

        results_lists["best_devel"]["trustworthiness"].append(results["trustworthiness"]["best_devel_ccc"])
        if configuration.are_test_labels_available:
            results_lists["test"]["trustworthiness"].append(results["trustworthiness"]["test_ccc"])

    with open(results["method_string"] + ".txt", "w") as fp:
        for d_ccc_a in zip(results_lists["best_devel"]["trustworthiness"]):
            fp.write(repr(d_ccc_a) + "\n")


def run_all():
    # TASK 3

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
    }, task=3)

run_all()
