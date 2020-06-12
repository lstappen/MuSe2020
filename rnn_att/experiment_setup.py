import os
import collections

import pandas as pd
import numpy as np
import tensorflow as tf

import core
import data_provider
from common import dict_to_struct, make_dirs_safe
from configuration import *


def get_partition(challenge_folder, name="partition"):
    names = os.listdir(challenge_folder + "/" + "label_segments/arousal")
    sample_ids = []
    for n in names:
        name_split = n[:-4]
        sample_ids.append(int(name_split))
    sample_ids = set(sample_ids)

    df = pd.read_csv(PARTITION_PROPOSAL_PATH, delimiter=",")
    data = df[["Id", "Proposal"]].values

    id_to_partition = dict()
    partition_to_id = collections.defaultdict(set)

    for i in range(data.shape[0]):
        sample_id = int(data[i, 0])
        partition = data[i, 1]

        if sample_id not in sample_ids:
            continue

        id_to_partition[sample_id] = partition
        partition_to_id[partition].add(sample_id)

    return id_to_partition, partition_to_id


def get_partition_to_chunk(partition_to_id, tf_records_folder):
    partition_to_chunk = collections.defaultdict(list)

    names = collections.defaultdict(list)
    names_unfiltered = os.listdir(tf_records_folder)
    for name in names_unfiltered:
        path_split = name.split(".")[0]
        path_split = path_split.split("_")
        names[int(path_split[0])].append(int(path_split[1]))

    for partition in partition_to_id.keys():
        for sample_id in partition_to_id[partition]:
            if sample_id in names.keys():
                for chunk_id in names[sample_id]:
                    partition_to_chunk[partition].append(repr(sample_id) + "_" + repr(chunk_id))

    return partition_to_chunk


def train(configuration):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = repr(configuration.GPU)

    ####################################################################################################################
    # Interpret configuration arguments.
    ####################################################################################################################
    are_test_labels_available = configuration.are_test_labels_available

    task_name = "task" + repr(configuration.task)

    id_to_partition, partition_to_id = get_partition(configuration.challenge_folder)

    partition_to_chunk = get_partition_to_chunk(partition_to_id, configuration.tf_records_folder)

    train_size = len(partition_to_chunk["train"])
    devel_size = len(partition_to_chunk["devel"])
    test_size = len(partition_to_chunk["test"])

    print(train_size, devel_size, test_size)

    train_steps_per_epoch = train_size // configuration.train_batch_size
    devel_steps_per_epoch = devel_size // configuration.devel_batch_size
    test_steps_per_epoch = test_size // configuration.test_batch_size

    print(train_steps_per_epoch, devel_steps_per_epoch, test_steps_per_epoch)

    tf_records_folder = configuration.tf_records_folder
    output_folder = configuration.output_folder
    make_dirs_safe(output_folder)

    if configuration.task == 1:
        targets = ["arousal", "valence"]
        number_of_targets = len(targets)
    elif configuration.task == 2:
        targets = ["arousal",
                   "valence",
                   "topic"]
        number_of_targets = 16
    elif configuration.task == 3:
        targets = ["arousal",
                   "valence",
                   "trustworthiness"]
        number_of_targets = len(targets)
    else:
        raise ValueError("Invalid task selection.")

    feature_names = [k for k, v in configuration.use_data.items() if v is True]

    method_string = "_" + repr(configuration.task) + "_" + \
                    "_".join([k for k, v in configuration.use_data.items() if v is True]) + "_" + \
                    configuration.model_type + "_" + \
                    repr(configuration.hidden_units) + "_" + \
                    repr(configuration.use_attention) + "_" + \
                    repr(configuration.initial_learning_rate) + "_" + \
                    repr(configuration.GPU)

    method_string = configuration.output_folder + "/" + method_string

    saver_paths = dict()
    for target in targets:
        saver_paths[target] = method_string + "_" + target + "_" + repr(configuration.GPU)

    ####################################################################################################################
    # Form computational graph.
    ####################################################################################################################
    g = tf.Graph()
    with g.as_default():
        with tf.Session() as sess:
            ############################################################################################################
            # Get dataset iterators.
            ############################################################################################################
            dataset_train = data_provider.get_split(tf_records_folder,
                                                    is_training=True,
                                                    task_name=task_name,
                                                    split_name="train",
                                                    are_test_labels_available=are_test_labels_available,
                                                    id_to_partition=id_to_partition,
                                                    feature_names=feature_names,
                                                    batch_size=configuration.train_batch_size,
                                                    seq_length=configuration.full_seq_length,
                                                    buffer_size=(train_steps_per_epoch + 1) // 4)
            dataset_devel = data_provider.get_split(tf_records_folder,
                                                    is_training=False,
                                                    task_name=task_name,
                                                    split_name="devel",
                                                    are_test_labels_available=are_test_labels_available,
                                                    id_to_partition=id_to_partition,
                                                    feature_names=feature_names,
                                                    batch_size=configuration.devel_batch_size,
                                                    seq_length=configuration.full_seq_length,
                                                    buffer_size=(devel_steps_per_epoch + 1) // 4)
            dataset_test = data_provider.get_split(tf_records_folder,
                                                   is_training=False,
                                                   task_name=task_name,
                                                   split_name="test",
                                                   are_test_labels_available=are_test_labels_available,
                                                   id_to_partition=id_to_partition,
                                                   feature_names=feature_names,
                                                   batch_size=configuration.test_batch_size,
                                                   seq_length=configuration.full_seq_length,
                                                   buffer_size=(test_steps_per_epoch + 1) // 4)

            iterator_train = tf.data.Iterator.from_structure(dataset_train.output_types,
                                                             dataset_train.output_shapes)
            iterator_devel = tf.data.Iterator.from_structure(dataset_devel.output_types,
                                                             dataset_devel.output_shapes)
            iterator_test = tf.data.Iterator.from_structure(dataset_test.output_types,
                                                            dataset_test.output_shapes)

            next_element_train = iterator_train.get_next()
            next_element_devel = iterator_devel.get_next()
            next_element_test = iterator_test.get_next()

            init_op_train = iterator_train.make_initializer(dataset_train)
            init_op_devel = iterator_devel.make_initializer(dataset_devel)
            init_op_test = iterator_test.make_initializer(dataset_test)

            ############################################################################################################
            # Define placeholders.
            ############################################################################################################
            batch_size_tensor = tf.placeholder(tf.int32)
            sequence_length_tensor = tf.placeholder(tf.int32)

            support_train = tf.placeholder(tf.float32, (None, None, 1))
            if task_name == "task2":
                topic_train = tf.placeholder(tf.float32, (None, 10))
            if task_name == "task3":
                trustworthiness_train = tf.placeholder(tf.float32, (None, None, 1))
            if task_name in ["task1", "task3"]:
                arousal_train = tf.placeholder(tf.float32, (None, None, 1))
                valence_train = tf.placeholder(tf.float32, (None, None, 1))
            elif task_name == "task2":
                arousal_train = tf.placeholder(tf.float32, (None, 3))
                valence_train = tf.placeholder(tf.float32, (None, 3))
            else:
                raise ValueError

            step_id_train = tf.placeholder(tf.int32, (None, None, 1))
            chunk_id_train = tf.placeholder(tf.int32, (None, None, 1))
            recording_id_train = tf.placeholder(tf.int32, (None, None, 1))

            tf_placeholder_train_dict = dict()
            for feature_name in feature_names:
                tf_placeholder_train_dict[feature_name] = tf.placeholder(tf.float32,
                                                                         (None,
                                                                          None,
                                                                          FEATURE_NUM[feature_name]))

            ############################################################################################################
            # Define model graph and get model.
            ############################################################################################################
            with tf.variable_scope("Model"):
                input_data_train = dict()
                for feature_name in feature_names:
                    if configuration.use_data[feature_name]:
                        input_data_train[feature_name] = tf_placeholder_train_dict[feature_name]

                if task_name in ["task1", "task3"]:
                    use_pooling = False
                elif task_name == "task2":
                    use_pooling = True
                else:
                    raise ValueError

                pred_train = core.get_model(input_data_dict=input_data_train,
                                            support=support_train,
                                            model_type=configuration.model_type,
                                            batch_size=batch_size_tensor,
                                            number_of_outputs=number_of_targets,
                                            orig_seq_length=sequence_length_tensor,
                                            hidden_units=configuration.hidden_units,
                                            use_attention=configuration.use_attention,
                                            use_pooling=use_pooling)

            ############################################################################################################
            # Define loss function.
            ############################################################################################################
            tensor_shape_train = [batch_size_tensor, sequence_length_tensor]
            flattened_size_train = tensor_shape_train[0] * tensor_shape_train[1]

            if task_name in ["task1", "task3"]:
                pred_arousal_train = pred_train[:, :, 0]
                pred_valence_train = pred_train[:, :, 1]
                if task_name == "task3":
                    pred_trustworthiness_train = pred_train[:, :, 2]

                single_pred_arousal_train = core.flatten_data(pred_arousal_train,
                                                              flattened_size_train)
                single_pred_valence_train = core.flatten_data(pred_valence_train,
                                                              flattened_size_train)
                if task_name == "task3":
                    single_pred_trustworthiness_train = core.flatten_data(pred_trustworthiness_train,
                                                                          flattened_size_train)

                single_true_support_train = core.flatten_data(support_train,
                                                              flattened_size_train)
                single_true_arousal_train = core.flatten_data(arousal_train,
                                                              flattened_size_train)
                single_true_valence_train = core.flatten_data(valence_train,
                                                              flattened_size_train)
                if task_name == "task3":
                    single_true_trustworthiness_train = core.flatten_data(trustworthiness_train,
                                                                          flattened_size_train)
            elif task_name == "task2":
                pred_arousal_train = pred_train[:, 0:3]
                pred_valence_train = pred_train[:, 3:6]
                pred_topic_train = pred_train[:, 6:16]

                single_pred_arousal_train = pred_arousal_train
                single_pred_valence_train = pred_valence_train
                single_pred_topic_train = pred_topic_train

                single_true_support_train = core.flatten_data(support_train,
                                                              flattened_size_train)
                single_true_arousal_train = arousal_train
                single_true_valence_train = valence_train
                single_true_topic_train = topic_train
            else:
                raise ValueError

            if task_name == "task1":
                loss = core.loss_function_task_1(pred_arousal=single_pred_arousal_train,
                                                 true_arousal=single_true_arousal_train,
                                                 pred_valence=single_pred_valence_train,
                                                 true_valence=single_true_valence_train,
                                                 support=single_true_support_train)
            elif task_name == "task2":
                loss = core.loss_function_task_2(pred_arousal=single_pred_arousal_train,
                                                 true_arousal=single_true_arousal_train,
                                                 pred_valence=single_pred_valence_train,
                                                 true_valence=single_true_valence_train,
                                                 pred_topic=single_pred_topic_train,
                                                 true_topic=single_true_topic_train,
                                                 support=single_true_support_train)
            elif task_name == "task3":
                loss = core.loss_function_task_3(pred_arousal=single_pred_arousal_train,
                                                 true_arousal=single_true_arousal_train,
                                                 pred_valence=single_pred_valence_train,
                                                 true_valence=single_true_valence_train,
                                                 pred_trustworthiness=single_pred_trustworthiness_train,
                                                 true_trustworthiness=single_true_trustworthiness_train,
                                                 support=single_true_support_train)
            else:
                raise NotImplementedError

            vars = tf.trainable_variables()
            model_vars = [v for v in vars if v.name.startswith("Model")]
            saver_dict = dict()

            for target in targets:
                saver_dict[target] = tf.train.Saver({v.name: v for v in model_vars})

            total_loss = tf.reduce_sum(loss)

            optimizer = tf.train.AdamOptimizer(configuration.initial_learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            optimizer = optimizer.apply_gradients(zip(gradients, variables))
            ############################################################################################################
            # Initialize variables and perform experiment.
            ############################################################################################################
            sess.run(tf.global_variables_initializer())

            ############################################################################################################
            # Train base model.
            ############################################################################################################
            current_patience = 0

            print("Start training base model.")
            print("Fresh base model.")
            for ee, epoch in enumerate(range(configuration.num_epochs)):
                print("EPOCH:", epoch + 1)

                input_feed_dict = {batch_size_tensor: "batch_size",
                                   sequence_length_tensor: "sequence_length",
                                   arousal_train: "arousal",
                                   valence_train: "valence",
                                   support_train: "support"
                                   }
                if task_name == "task2":
                    input_feed_dict[topic_train] = "topic"
                if task_name == "task3":
                    input_feed_dict[trustworthiness_train] = "trustworthiness"
                for feature_name in feature_names:
                    input_feed_dict[tf_placeholder_train_dict[feature_name]] = feature_name

                run_epoch = core.RunEpoch(sess=sess,
                                          partition="train",
                                          are_test_labels_available=are_test_labels_available,
                                          init_op=init_op_train,
                                          steps_per_epoch=train_steps_per_epoch,
                                          next_element=next_element_train,
                                          batch_size=configuration.train_batch_size,
                                          seq_length=configuration.full_seq_length,
                                          input_gaussian_noise=configuration.input_gaussian_noise,
                                          optimizer=optimizer,
                                          loss=total_loss,
                                          pred=pred_train,
                                          input_feed_dict=input_feed_dict,
                                          targets=targets,
                                          task_name=task_name)

                train_items, train_subject_to_id = run_epoch.run_epoch()

                if task_name == "task1":
                    train_measures = core.get_measures_task_1(train_items)
                elif task_name == "task2":
                    train_measures = core.get_measures_task_2(train_items)
                elif task_name == "task3":
                    train_measures = core.get_measures_task_3(train_items)
                else:
                    raise NotImplementedError
                print(method_string)

                if task_name == "task1":
                    print("Train CCC:", train_measures["arousal"]["ccc"],
                                        train_measures["valence"]["ccc"],
                                        train_measures["ccc"])

                    print("Train  CC:", train_measures["arousal"]["cc"],
                                        train_measures["valence"]["cc"],
                                        train_measures["cc"])

                    print("Train MAE:", train_measures["arousal"]["mae"],
                                        train_measures["valence"]["mae"],
                                        train_measures["mae"])
                elif task_name == "task2":
                    print("Train UAR:", train_measures["arousal"]["macro-recall"],
                                        train_measures["valence"]["macro-recall"],
                                        train_measures["topic"]["macro-recall"])

                    print("Train  F1:", train_measures["arousal"]["micro-f1"],
                                        train_measures["valence"]["micro-f1"],
                                        train_measures["topic"]["micro-f1"])

                    print("Train TOT:", train_measures["arousal"]["score"],
                                        train_measures["valence"]["score"],
                                        train_measures["topic"]["score"])
                elif task_name == "task3":
                    print("Train CCC:", train_measures["trustworthiness"]["ccc"])

                    print("Train  CC:", train_measures["trustworthiness"]["cc"])

                    print("Train MAE:", train_measures["trustworthiness"]["mae"])
                else:
                    raise NotImplementedError

                if ee == 0:
                    best_performance_dict = dict()
                    if task_name == "task1":
                        for target in targets:
                            best_performance_dict[target] = -1.0
                    elif task_name == "task2":
                        for target in targets:
                            best_performance_dict[target] = dict()
                            for measure_name in ["macro-recall", "micro-f1", "score"]:
                                best_performance_dict[target][measure_name] = -1.0
                    elif task_name == "task3":
                        best_performance_dict["trustworthiness"] = -1.0
                    else:
                        raise NotImplementedError

                if (ee) % configuration.val_every_n_epoch == 0:

                    input_feed_dict = {batch_size_tensor: "batch_size",
                                       sequence_length_tensor: "sequence_length",
                                       arousal_train: "arousal",
                                       valence_train: "valence",
                                       support_train: "support"
                                       }
                    if task_name == "task2":
                        input_feed_dict[topic_train] = "topic"
                    if task_name == "task3":
                        input_feed_dict[trustworthiness_train] = "trustworthiness"
                    for feature_name in feature_names:
                        input_feed_dict[tf_placeholder_train_dict[feature_name]] = feature_name

                    run_epoch = core.RunEpoch(sess=sess,
                                              partition="devel",
                                              are_test_labels_available=are_test_labels_available,
                                              init_op=init_op_devel,
                                              steps_per_epoch=devel_steps_per_epoch,
                                              next_element=next_element_devel,
                                              batch_size=configuration.devel_batch_size,
                                              seq_length=configuration.full_seq_length,
                                              input_gaussian_noise=configuration.input_gaussian_noise,
                                              optimizer=None,
                                              loss=None,
                                              pred=pred_train,
                                              input_feed_dict=input_feed_dict,
                                              targets=targets,
                                              task_name=task_name)

                    devel_items, devel_subject_to_id = run_epoch.run_epoch()

                    if task_name == "task1":
                        devel_measures = core.get_measures_task_1(devel_items)
                    elif task_name == "task2":
                        devel_measures = core.get_measures_task_2(devel_items)
                    elif task_name == "task3":
                        devel_measures = core.get_measures_task_3(devel_items)
                    else:
                        raise NotImplementedError

                    if task_name == "task1":
                        print("Devel CCC:", devel_measures["arousal"]["ccc"],
                                            devel_measures["valence"]["ccc"],
                                            devel_measures["ccc"])

                        print("Devel  CC:", devel_measures["arousal"]["cc"],
                                            devel_measures["valence"]["cc"],
                                            devel_measures["cc"])

                        print("Devel MAE:", devel_measures["arousal"]["mae"],
                                            devel_measures["valence"]["mae"],
                                            devel_measures["mae"])
                    elif task_name == "task2":
                        print("Devel UAR:", devel_measures["arousal"]["macro-recall"],
                                            devel_measures["valence"]["macro-recall"],
                                            devel_measures["topic"]["macro-recall"])

                        print("Devel  F1:", devel_measures["arousal"]["micro-f1"],
                                            devel_measures["valence"]["micro-f1"],
                                            devel_measures["topic"]["micro-f1"])

                        print("Devel TOT:", devel_measures["arousal"]["score"],
                                            devel_measures["valence"]["score"],
                                            devel_measures["topic"]["score"])
                    elif task_name == "task3":
                        print("Devel CCC:", devel_measures["trustworthiness"]["ccc"])

                        print("Devel  CC:", devel_measures["trustworthiness"]["cc"])

                        print("Devel MAE:", devel_measures["trustworthiness"]["mae"])
                    else:
                        raise NotImplementedError

                    noticed_improvement = False

                    if task_name == "task1":
                        for target in targets:
                            if best_performance_dict[target] < devel_measures[target]["ccc"]:
                                best_performance_dict[target] = devel_measures[target]["ccc"]
                                saver_dict[target].save(sess, saver_paths[target])
                                noticed_improvement = True
                    elif task_name == "task2":
                        for target in targets:
                            if best_performance_dict[target]["score"] < devel_measures[target]["score"]:
                                for measure_name in ["macro-recall", "micro-f1", "score"]:
                                    best_performance_dict[target][measure_name] = devel_measures[target][measure_name]
                                saver_dict[target].save(sess, saver_paths[target])
                                noticed_improvement = True
                    elif task_name == "task3":
                        if best_performance_dict["trustworthiness"] < devel_measures["trustworthiness"]["ccc"]:
                            best_performance_dict["trustworthiness"] = devel_measures["trustworthiness"]["ccc"]
                            saver_dict["trustworthiness"].save(sess, saver_paths["trustworthiness"])
                            noticed_improvement = True
                    else:
                        raise NotImplementedError

                    if noticed_improvement:
                        current_patience = 0
                    else:
                        current_patience += 1
                        if current_patience > configuration.patience:
                            break

                else:
                    pass

            test_measures_dict = dict()
            test_items_dict = dict()
            for target in targets:
                if task_name == "task3":
                    if target not in ["trustworthiness", ]:
                        continue
                saver_dict[target].restore(sess, saver_paths[target])

                input_feed_dict = {batch_size_tensor: "batch_size",
                                   sequence_length_tensor: "sequence_length",
                                   arousal_train: "arousal",
                                   valence_train: "valence",
                                   support_train: "support"
                                   }
                if task_name == "task2":
                    input_feed_dict[topic_train] = "topic"
                if task_name == "task3":
                    input_feed_dict[trustworthiness_train] = "trustworthiness"
                for feature_name in feature_names:
                    input_feed_dict[tf_placeholder_train_dict[feature_name]] = feature_name

                run_epoch = core.RunEpoch(sess=sess,
                                          partition="test",
                                          are_test_labels_available=are_test_labels_available,
                                          init_op=init_op_test,
                                          steps_per_epoch=test_steps_per_epoch,
                                          next_element=next_element_test,
                                          batch_size=configuration.test_batch_size,
                                          seq_length=configuration.full_seq_length,
                                          input_gaussian_noise=configuration.input_gaussian_noise,
                                          optimizer=None,
                                          loss=None,
                                          pred=pred_train,
                                          input_feed_dict=input_feed_dict,
                                          targets=targets,
                                          task_name=task_name)

                test_items, test_subject_to_id = run_epoch.run_epoch()

                if are_test_labels_available:
                    if task_name == "task1":
                        test_measures = core.get_measures_task_1(test_items)
                    elif task_name == "task2":
                        test_measures = core.get_measures_task_2(test_items)
                    elif task_name == "task3":
                        test_measures = core.get_measures_task_3(test_items)
                    else:
                        raise NotImplementedError

                    test_measures_dict[target] = test_measures
                test_items_dict[target] = test_items

            if task_name == "task1":
                print("Best devel CCC:",
                      best_performance_dict["arousal"],
                      best_performance_dict["valence"],
                      (best_performance_dict["arousal"] + best_performance_dict["valence"]) / 2.0)

                if are_test_labels_available:
                    print("Test CCC:", test_measures_dict["arousal"]["arousal"]["ccc"],
                          test_measures_dict["valence"]["valence"]["ccc"],
                          (test_measures_dict["arousal"]["arousal"]["ccc"] + test_measures_dict["valence"]["valence"][
                              "ccc"]) / 2.0)

                    print("Test  CC:", test_measures_dict["arousal"]["arousal"]["cc"],
                          test_measures_dict["valence"]["valence"]["cc"],
                          (test_measures_dict["arousal"]["arousal"]["cc"] + test_measures_dict["valence"]["valence"][
                              "cc"]) / 2.0)

                    print("Test MAE:", test_measures_dict["arousal"]["arousal"]["mae"],
                          test_measures_dict["valence"]["valence"]["mae"],
                          (test_measures_dict["arousal"]["arousal"]["mae"] + test_measures_dict["valence"]["valence"][
                              "mae"]) / 2.0)
            elif task_name == "task2":
                print("Best devel CCC:",
                      best_performance_dict["arousal"]["score"],
                      best_performance_dict["valence"]["score"],
                      (best_performance_dict["arousal"]["score"] + best_performance_dict["valence"]["score"]) / 2.0,
                      best_performance_dict["topic"]["score"])

                if are_test_labels_available:
                    print("Test  UAR:", test_measures_dict["arousal"]["arousal"]["macro-recall"],
                          test_measures_dict["valence"]["valence"]["macro-recall"],
                          test_measures_dict["topic"]["topic"]["macro-recall"])

                    print("Test   F1:", test_measures_dict["arousal"]["arousal"]["micro-f1"],
                          test_measures_dict["valence"]["valence"]["micro-f1"],
                          test_measures_dict["topic"]["topic"]["micro-f1"])

                    print("Test  TOT:", 0.66 * test_measures_dict["arousal"]["arousal"]["micro-f1"] + 0.34 *
                          test_measures_dict["arousal"]["arousal"]["macro-recall"],
                          0.66 * test_measures_dict["valence"]["valence"]["micro-f1"] + 0.34 *
                          test_measures_dict["valence"]["valence"]["macro-recall"],
                          0.66 * test_measures_dict["topic"]["topic"]["micro-f1"] + 0.34 *
                          test_measures_dict["topic"]["topic"]["macro-recall"])
            elif task_name == "task3":
                print("Best devel CCC:",
                      best_performance_dict["trustworthiness"])

                if are_test_labels_available:
                    print("Test CCC:", test_measures_dict["trustworthiness"]["trustworthiness"]["ccc"])

                    print("Test  CC:", test_measures_dict["trustworthiness"]["trustworthiness"]["cc"])

                    print("Test MAE:", test_measures_dict["trustworthiness"]["trustworthiness"]["mae"])
            else:
                raise NotImplementedError

            if task_name == "task1":
                results = dict()
                results["method_string"] = method_string
                results["arousal"] = dict()
                results["valence"] = dict()
                results["arousal"]["best_devel_ccc"] = best_performance_dict["arousal"]
                results["valence"]["best_devel_ccc"] = best_performance_dict["valence"]
                if are_test_labels_available:
                    results["arousal"]["test_ccc"] = test_measures_dict["arousal"]["arousal"]["ccc"]
                    results["valence"]["test_ccc"] = test_measures_dict["valence"]["valence"]["ccc"]
                    results["arousal"]["test_cc"] = test_measures_dict["arousal"]["arousal"]["cc"]
                    results["valence"]["test_cc"] = test_measures_dict["valence"]["valence"]["cc"]
                    results["arousal"]["test_mae"] = test_measures_dict["arousal"]["arousal"]["mae"]
                    results["valence"]["test_mae"] = test_measures_dict["valence"]["valence"]["mae"]
                results["arousal"]["test_true"] = test_items_dict["arousal"].arousal.true
                results["valence"]["test_true"] = test_items_dict["valence"].valence.true
                results["arousal"]["test_pred"] = test_items_dict["arousal"].arousal.pred
                results["valence"]["test_pred"] = test_items_dict["valence"].valence.pred

                print("Saving test predictions at:", method_string)
                np.save(method_string + "/arousal_test_pred.npy", test_items_dict["arousal"].arousal.pred)
                np.save(method_string + "/valence_test_pred.npy", test_items_dict["valence"].valence.pred)
            elif task_name == "task2":
                results = dict()
                results["method_string"] = method_string
                results["arousal"] = dict()
                results["valence"] = dict()
                results["emotion"] = dict()
                results["topic"] = dict()

                results["arousal"]["best_devel_macro_recall"] = best_performance_dict["arousal"]["macro-recall"]
                results["arousal"]["best_devel_micro_f1"] = best_performance_dict["arousal"]["micro-f1"]
                results["arousal"]["best_devel_score"] = best_performance_dict["arousal"]["score"]
                results["valence"]["best_devel_macro_recall"] = best_performance_dict["valence"]["macro-recall"]
                results["valence"]["best_devel_micro_f1"] = best_performance_dict["valence"]["micro-f1"]
                results["valence"]["best_devel_score"] = best_performance_dict["valence"]["score"]
                results["emotion"]["best_devel_macro_recall"] = (best_performance_dict["arousal"]["macro-recall"] +
                                                          best_performance_dict["valence"]["macro-recall"]) / 2.0
                results["emotion"]["best_devel_micro_f1"] = (best_performance_dict["arousal"]["micro-f1"] +
                                                          best_performance_dict["valence"]["micro-f1"]) / 2.0
                results["emotion"]["best_devel_score"] = (best_performance_dict["arousal"]["score"] +
                                                          best_performance_dict["valence"]["score"]) / 2.0
                results["topic"]["best_devel_macro_recall"] = best_performance_dict["topic"]["macro-recall"]
                results["topic"]["best_devel_micro_f1"] = best_performance_dict["topic"]["micro-f1"]
                results["topic"]["best_devel_score"] = best_performance_dict["topic"]["score"]
                if are_test_labels_available:
                    results["arousal"]["test_macro_recall"] = test_measures_dict["arousal"]["arousal"]["macro-recall"]
                    results["valence"]["test_macro_recall"] = test_measures_dict["valence"]["valence"]["macro-recall"]
                    results["emotion"]["test_macro_recall"] = (test_measures_dict["arousal"]["arousal"][
                                                                   "macro-recall"] +
                                                               test_measures_dict["valence"]["valence"][
                                                                   "macro-recall"]) / 2.0
                    results["topic"]["test_macro_recall"] = test_measures_dict["valence"]["valence"]["macro-recall"]

                    results["arousal"]["test_micro_f1"] = test_measures_dict["arousal"]["arousal"]["micro-f1"]
                    results["valence"]["test_micro_f1"] = test_measures_dict["valence"]["valence"]["micro-f1"]
                    results["emotion"]["test_micro_f1"] = (test_measures_dict["arousal"]["arousal"]["micro-f1"] +
                                                           test_measures_dict["valence"]["valence"]["micro-f1"]) / 2.0
                    results["topic"]["test_micro_f1"] = test_measures_dict["valence"]["valence"]["micro-f1"]

                    results["arousal"]["test_score"] = 0.66 * results["arousal"]["test_micro_f1"] + 0.34 * \
                                                       results["arousal"]["test_macro_recall"]
                    results["valence"]["test_score"] = 0.66 * results["valence"]["test_micro_f1"] + 0.34 * \
                                                       results["valence"]["test_macro_recall"]
                    results["emotion"]["test_score"] = (results["arousal"]["test_score"] +
                                                        results["valence"]["test_score"]) / 2.0
                    results["topic"]["test_score"] = 0.66 * results["topic"]["test_micro_f1"] + 0.34 * results["topic"][
                        "test_macro_recall"]

                results["arousal"]["test_true"] = test_items_dict["arousal"].arousal.true
                results["valence"]["test_true"] = test_items_dict["valence"].valence.true
                results["topic"]["test_true"] = test_items_dict["topic"].topic.true
                results["arousal"]["test_pred"] = test_items_dict["arousal"].arousal.pred
                results["valence"]["test_pred"] = test_items_dict["valence"].valence.pred
                results["topic"]["test_pred"] = test_items_dict["topic"].topic.pred

                print("Saving test predictions at:", method_string)
                np.save(method_string + "/arousal_test_pred.npy", test_items_dict["arousal"].arousal.pred)
                np.save(method_string + "/valence_test_pred.npy", test_items_dict["valence"].valence.pred)
                np.save(method_string + "/topic_test_pred.npy", test_items_dict["topic"].topic.pred)
            elif task_name == "task3":
                results = dict()
                results["method_string"] = method_string
                results["trustworthiness"] = dict()
                results["trustworthiness"]["best_devel_ccc"] = best_performance_dict["trustworthiness"]
                if are_test_labels_available:
                    results["trustworthiness"]["test_ccc"] = test_measures_dict["trustworthiness"]["trustworthiness"][
                        "ccc"]
                    results["trustworthiness"]["test_cc"] = test_measures_dict["trustworthiness"]["trustworthiness"][
                        "cc"]
                    results["trustworthiness"]["test_mae"] = test_measures_dict["trustworthiness"]["trustworthiness"][
                        "mae"]
                results["trustworthiness"]["test_true"] = test_items_dict["trustworthiness"].arousal.true
                results["trustworthiness"]["test_pred"] = test_items_dict["trustworthiness"].arousal.pred

                print("Saving test predictions at:", method_string)
                np.save(method_string + "/trustworthiness_test_pred.npy", test_items_dict["trustworthiness"].trustworthiness.pred)
            else:
                raise NotImplementedError

            return results
