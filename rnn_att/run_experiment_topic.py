from experiment_setup import train
from common import dict_to_struct

from configuration import *
#TODO: Models with masked test labels

def run(features_dict, task):
    results_lists = dict()
    results_lists["best_devel"] = dict()
    results_lists["test"] = dict()

    for target in ["arousal", "valence", "emotion", "topic"]:
        results_lists["best_devel"][target] = dict()
        results_lists["test"][target] = dict()
        for measure in ["macro_recall", "micro_f1", "score"]:
            results_lists["best_devel"][target][measure] = list()
            results_lists["test"][target][measure] = list()

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
        configuration["model_type"] = "lstm"  # ["2dcnn_lstm", "lstm"]

        configuration["input_gaussian_noise"] = 0.1
        configuration["hidden_units"] = 40
        configuration["use_attention"] = True
        configuration["initial_learning_rate"] = 0.0001
        configuration["full_seq_length"] = SEQ_LEN
        configuration["train_batch_size"] = 20
        configuration["devel_batch_size"] = 20
        configuration["test_batch_size"] = 20
        configuration["num_epochs"] = 15
        configuration["patience"] = 5
        configuration["val_every_n_epoch"] = 1

        configuration["GPU"] = 0

        configuration = dict_to_struct(configuration)

        results = train(configuration)

        for target in ["arousal", "valence", "emotion", "topic"]:
            for measure in ["macro_recall", "micro_f1", "score"]:
                results_lists["best_devel"][target][measure].append(results[target]["best_devel_" + measure])
                if configuration.are_test_labels_available:
                    results_lists["test"][target][measure].append(results[target]["test_" + measure])

    with open(results["method_string"] + ".txt", "w") as fp:
        for d_macro_recall_a, d_macro_recall_v, d_macro_recall_e, d_macro_recall_t, \
            d_micro_f1_a, d_micro_f1_v, d_micro_f1_e, d_micro_f1_t, \
            d_score_a, d_score_v, d_score_e, d_score_t in zip(results_lists["best_devel"]["arousal"]["macro_recall"],
                                                              results_lists["best_devel"]["valence"]["macro_recall"],
                                                              results_lists["best_devel"]["emotion"]["macro_recall"],
                                                              results_lists["best_devel"]["topic"]["macro_recall"],
                                                              results_lists["best_devel"]["arousal"]["micro_f1"],
                                                              results_lists["best_devel"]["valence"]["micro_f1"],
                                                              results_lists["best_devel"]["emotion"]["micro_f1"],
                                                              results_lists["best_devel"]["topic"]["micro_f1"],
                                                              results_lists["best_devel"]["arousal"]["score"],
                                                              results_lists["best_devel"]["valence"]["score"],
                                                              results_lists["best_devel"]["emotion"]["score"],
                                                              results_lists["best_devel"]["topic"]["score"]):
            fp.write(repr(d_macro_recall_a) + "\t" +
                     repr(d_macro_recall_v) + "\t" +
                     repr(d_macro_recall_e) + "\t" +
                     repr(d_macro_recall_t) + "\t" +
                     repr(d_micro_f1_a) + "\t" +
                     repr(d_micro_f1_v) + "\t" +
                     repr(d_micro_f1_e) + "\t" +
                     repr(d_micro_f1_t) + "\t" +
                     repr(d_score_a) + "\t" +
                     repr(d_score_v) + "\t" +
                     repr(d_score_e) + "\t" +
                     repr(d_score_t) + "\n")


def run_all():
    run({
        "au": False,
        "deepspectrum": True,
        "egemaps": False,
        "fasttext": False,
        "gaze": False,
        "gocar": False,
        "landmarks_2d": False,
        "landmarks_3d": False,
        "openpose": False,
        "pdm": False,
        "pose": False,
        "vggface": False,
        "xception": False
    }, task=2)

run_all()
