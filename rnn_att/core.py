import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Attention
import pandas as pd
from sklearn.metrics import mean_absolute_error, f1_score, recall_score


from common import dict_to_struct


def concat_input_data(input_data_list):
    return tf.concat(input_data_list, axis=2)


def get_model(input_data_dict,
              support,
              model_type,
              batch_size,
              number_of_outputs,
              orig_seq_length,
              hidden_units,
              use_attention,
              use_pooling):
    with tf.variable_scope("Fusion"):
        net = concat_input_data([v for v in input_data_dict.values()])

    _, seq_length, num_features = net.get_shape().as_list()

    if model_type == "lstm":
        net = get_lstm_model(net,
                             batch_size,
                             number_of_outputs,
                             orig_seq_length,
                             support,
                             1,
                             hidden_units,
                             use_attention,
                             None,
                             use_pooling)
        _, _, num_features, _ = net.get_shape().as_list()

        if use_pooling:
            net = tf.reshape(net, (batch_size, num_features))
        else:
            net = tf.reshape(net, (batch_size, orig_seq_length, num_features))
    else:
        raise ValueError("Invalid model type.")

    prediction = net

    return prediction


def get_lstm_model(input_data,
                   batch_size,
                   number_of_outputs,
                   orig_seq_length,
                   support,
                   num_layers,
                   hidden_units,
                   use_attention,
                   activation,
                   use_pooling):
    _, _, num_features = input_data.get_shape().as_list()

    seq_length_for_rnn = tf.reshape(support, (batch_size, -1))
    seq_length_for_rnn = tf.cast(tf.reduce_sum(seq_length_for_rnn, axis=1), tf.int32)

    def _get_cell(l_no, counter_mutable):
        lstm = tf.contrib.rnn.LSTMCell(hidden_units,
                                       use_peepholes=True,
                                       cell_clip=100,
                                       state_is_tuple=True,
                                       name="rnn_" + repr(counter_mutable[0]))
        counter_mutable[0] += 1
        return lstm

    counter_mutable = [0,]

    q_stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell([_get_cell(l_no, counter_mutable) for l_no in range(num_layers)], state_is_tuple=True)
    q_stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell([_get_cell(l_no, counter_mutable) for l_no in range(num_layers)], state_is_tuple=True)

    q_outputs, _ = tf.nn.bidirectional_dynamic_rnn(q_stacked_lstm_fw,
                                                   q_stacked_lstm_bw,
                                                   tf.reshape(input_data, (batch_size, orig_seq_length, num_features)),
                                                   sequence_length=seq_length_for_rnn,
                                                   dtype=tf.float32)

    q_outputs = q_outputs[0] + q_outputs[1]

    if use_attention:

        v_stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell([_get_cell(l_no, counter_mutable) for l_no in range(num_layers)],
                                                        state_is_tuple=True)
        v_stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell([_get_cell(l_no, counter_mutable) for l_no in range(num_layers)],
                                                        state_is_tuple=True)

        v_outputs, _ = tf.nn.bidirectional_dynamic_rnn(v_stacked_lstm_fw,
                                                       v_stacked_lstm_bw,
                                                       tf.reshape(input_data, (batch_size, orig_seq_length, num_features)),
                                                       sequence_length=seq_length_for_rnn,
                                                       dtype=tf.float32)

        v_outputs = v_outputs[0] + v_outputs[1]

        query_value_attention_seq = Attention()(
            [q_outputs, v_outputs])

        outputs = tf.concat([q_outputs, query_value_attention_seq], axis=2)

        net = tf.reshape(outputs, (batch_size * orig_seq_length, 2 * hidden_units))
    else:
        outputs = q_outputs
        net = tf.reshape(outputs, (batch_size * orig_seq_length, hidden_units))

    if use_pooling:
        _, num_features = net.get_shape().as_list()
        net = tf.reshape(net, (batch_size, orig_seq_length, num_features))
        net = tf.reduce_max(net, axis=1, keepdims=False)

        prediction = tf.layers.dense(net, number_of_outputs)
        if activation is not None:
            prediction = tf.nn.relu(prediction)

        prediction = tf.reshape(prediction, (batch_size, 1, number_of_outputs, 1))
    else:
        prediction = tf.layers.dense(net, number_of_outputs)
        if activation is not None:
            prediction = tf.nn.relu(prediction)

        prediction = tf.reshape(prediction, (batch_size, orig_seq_length, number_of_outputs, 1))

    return prediction


def flatten_data(data, flattened_size):
    flattened_data = tf.reshape(data[:, :],
                                (-1,))
    flattened_data = tf.reshape(flattened_data,
                                (flattened_size, 1, 1, 1))
    return flattened_data


def loss_function_task_1(pred_arousal,
                         true_arousal,
                         pred_valence,
                         true_valence,
                         support):
    loss = (weighted_concordance_cc(pred_arousal, support, true_arousal) +
            weighted_concordance_cc(pred_valence, support, true_valence)) / 2.0

    return loss


def loss_function_task_2(pred_arousal,
                         true_arousal,
                         pred_valence,
                         true_valence,
                         pred_topic,
                         true_topic,
                         support):
    loss = (tf.nn.softmax_cross_entropy_with_logits(labels=true_arousal,
                                                    logits=pred_arousal) +
            tf.nn.softmax_cross_entropy_with_logits(labels=true_valence,
                                                    logits=pred_valence) +
            tf.nn.softmax_cross_entropy_with_logits(labels=true_topic,
                                                    logits=pred_topic)
            ) / 3.0

    return loss


def loss_function_task_3(pred_arousal,
                         true_arousal,
                         pred_valence,
                         true_valence,
                         pred_trustworthiness,
                         true_trustworthiness,
                         support):
    loss = weighted_concordance_cc(pred_trustworthiness, support, true_trustworthiness)

    return loss


def weighted_concordance_cc(pred, support, true):
    mu_x = weighted_mean(pred, support)
    mu_y = weighted_mean(true, support)

    mean_cent_prod = weighted_covariance(pred, true, mu_x, mu_y, support)
    denom = weighted_covariance(pred, pred, mu_x, mu_x, support) + \
            weighted_covariance(true, true, mu_y, mu_y, support) + \
            tf.pow((mu_x - mu_y), 2)

    return 1.0 - (2.0 * mean_cent_prod) / denom


def weighted_mean(x, w):
    mu = tf.reduce_sum(tf.multiply(x, w)) / tf.reduce_sum(w)
    return mu


def weighted_covariance(x, y, mu_x, mu_y, w):
    sigma = tf.reduce_sum(tf.multiply(w, tf.multiply(x - mu_x, y - mu_y))) / tf.reduce_sum(w)
    return sigma


def make_sequence_list(data, support):
    sequence = list()
    for i in range(data.shape[0]):
        data_to_input = data[i, support[i, :] == 1.0].reshape(-1,)
        sequence.append(data_to_input)

    sequence = np.concatenate(sequence)
    return sequence


def get_measures_task_1(items):
    arousal_y_true = make_sequence_list(items.arousal.true, items.support)
    arousal_y_pred = make_sequence_list(items.arousal.pred, items.support)

    valence_y_true = make_sequence_list(items.valence.true, items.support)
    valence_y_pred = make_sequence_list(items.valence.pred, items.support)

    flat_ccc_arousal, flat_cc_arousal, flat_mae_arousal = flatten_score(arousal_y_true, arousal_y_pred)
    flat_ccc_valence, flat_cc_valence, flat_mae_valence = flatten_score(valence_y_true, valence_y_pred)

    measures = dict()

    measures["arousal"] = dict()
    measures["valence"] = dict()

    measures["arousal"]["ccc"] = flat_ccc_arousal
    measures["valence"]["ccc"] = flat_ccc_valence
    measures["ccc"] = (flat_ccc_arousal + flat_ccc_valence) / 2.0

    measures["arousal"]["cc"] = flat_cc_arousal
    measures["valence"]["cc"] = flat_cc_valence
    measures["cc"] = (flat_cc_arousal + flat_cc_valence) / 2.0

    measures["arousal"]["mae"] = flat_mae_arousal
    measures["valence"]["mae"] = flat_mae_valence
    measures["mae"] = (flat_mae_arousal + flat_mae_valence) / 2.0

    return measures


def get_measures_task_2(items):
    macro_recall_arousal = recall_score(y_true=np.argmax(items.arousal.true, axis=1),
                                        y_pred=np.argmax(items.arousal.pred, axis=1),
                                        average="macro")
    micro_f1_arousal = f1_score(y_true=np.argmax(items.arousal.true, axis=1),
                                y_pred=np.argmax(items.arousal.pred, axis=1),
                                average="micro")

    macro_recall_valence = recall_score(y_true=np.argmax(items.valence.true, axis=1),
                                        y_pred=np.argmax(items.valence.pred, axis=1),
                                        average="macro")
    micro_f1_valence = f1_score(y_true=np.argmax(items.valence.true, axis=1),
                                y_pred=np.argmax(items.valence.pred, axis=1),
                                average="micro")

    macro_recall_topic = recall_score(y_true=np.argmax(items.topic.true, axis=1),
                                      y_pred=np.argmax(items.topic.pred, axis=1),
                                      average="macro")
    micro_f1_topic = f1_score(y_true=np.argmax(items.topic.true, axis=1),
                              y_pred=np.argmax(items.topic.pred, axis=1),
                              average="micro")

    measures = dict()

    measures["arousal"] = dict()
    measures["arousal"]["macro-recall"] = macro_recall_arousal
    measures["arousal"]["micro-f1"] = micro_f1_arousal
    measures["arousal"]["score"] = 0.66 * micro_f1_arousal + 0.34 * macro_recall_arousal

    measures["valence"] = dict()
    measures["valence"]["macro-recall"] = macro_recall_valence
    measures["valence"]["micro-f1"] = micro_f1_valence
    measures["valence"]["score"] = 0.66 * micro_f1_valence + 0.34 * macro_recall_valence

    measures["emotion"] = dict()
    measures["emotion"]["macro-recall"] = (macro_recall_arousal + macro_recall_valence) / 2.0
    measures["emotion"]["micro-f1"] = (micro_f1_arousal + micro_f1_valence) / 2.0
    measures["emotion"]["score"] = (measures["arousal"]["score"] + measures["valence"]["score"]) / 2.0

    measures["topic"] = dict()
    measures["topic"]["macro-recall"] = macro_recall_topic
    measures["topic"]["micro-f1"] = micro_f1_topic
    measures["topic"]["score"] = 0.66 * micro_f1_topic + 0.34 * macro_recall_topic

    return measures


def get_measures_task_3(items):
    trustworthiness_y_true = make_sequence_list(items.trustworthiness.true, items.support)
    trustworthiness_y_pred = make_sequence_list(items.trustworthiness.pred, items.support)

    flat_ccc_trustworthiness, flat_cc_trustworthiness, flat_mae_trustworthiness = flatten_score(trustworthiness_y_true, trustworthiness_y_pred)

    measures = dict()

    measures["trustworthiness"] = dict()

    measures["trustworthiness"]["ccc"] = flat_ccc_trustworthiness

    measures["trustworthiness"]["cc"] = flat_cc_trustworthiness

    measures["trustworthiness"]["mae"] = flat_mae_trustworthiness

    return measures


def replace_dict_value(input_dict, old_value, new_value):
    for k, v in input_dict.items():
        if isinstance(v, str):
            if v == old_value:
                input_dict[k] = np.nan_to_num(new_value, copy=True)
    return input_dict


target_to_classdim = {"arousal": 3,
                      "valence": 3,
                      "topic": 10}


class RunEpoch:
    def __init__(self,
                 sess,
                 partition,
                 are_test_labels_available,
                 init_op,
                 steps_per_epoch,
                 next_element,
                 batch_size,
                 seq_length,
                 input_gaussian_noise,
                 optimizer,
                 loss,
                 pred,
                 input_feed_dict,
                 targets,
                 task_name):
        self.sess = sess
        self.partition = partition
        self.are_test_labels_available = are_test_labels_available
        self.init_op = init_op
        self.steps_per_epoch = steps_per_epoch
        self.next_element = next_element
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.optimizer = optimizer
        self.loss = loss
        self.pred = pred
        self.input_gaussian_noise = input_gaussian_noise
        self.input_feed_dict = input_feed_dict
        self.targets = targets
        self.task_name = task_name

        self.number_of_targets = len(self.targets)

    def run_epoch(self):
        batch_size_sum = 0

        # Initialize an iterator over the dataset split.
        self.sess.run(self.init_op)

        # Store variable sequence.
        stored_variables = dict()

        if self.task_name in ["task1", "task3"]:
            for target in self.targets:
                stored_variables[target] = dict()
                if not ((self.partition == "test") and (self.are_test_labels_available)):
                    stored_variables[target]["true"] = np.empty((self.steps_per_epoch * self.batch_size,
                                                                 self.seq_length),
                                                                dtype=np.float32)

                stored_variables[target]["pred"] = np.empty((self.steps_per_epoch * self.batch_size,
                                                             self.seq_length),
                                                            dtype=np.float32)

        elif self.task_name == "task2":
            for target in self.targets:
                stored_variables[target] = dict()
                if not ((self.partition == "test") and (self.are_test_labels_available)):
                    stored_variables[target]["true"] = np.empty((self.steps_per_epoch * self.batch_size,
                                                                 target_to_classdim[target]),
                                                                dtype=np.float32)

                stored_variables[target]["pred"] = np.empty((self.steps_per_epoch * self.batch_size,
                                                             target_to_classdim[target]),
                                                            dtype=np.float32)

        else:
            raise ValueError
        if self.task_name in ["task1", "task3"]:
            stored_variables["support"] = np.empty((self.steps_per_epoch * self.batch_size,
                                                    50),
                                                   dtype=np.float32)
        else:
            stored_variables["support"] = np.empty((self.steps_per_epoch * self.batch_size,
                                                    500),
                                                   dtype=np.float32)

        stored_variables["loss"] = None

        # Run epoch.
        subject_to_id = dict()
        for step in range(self.steps_per_epoch):
            batch_tuple = self.sess.run(self.next_element)
            step_id = batch_tuple["step_id"]
            chunk_id = batch_tuple["chunk_id"]
            recording_id = batch_tuple["recording_id"]

            au = np.nan_to_num(batch_tuple["au"])
            deepspectrum = np.nan_to_num(batch_tuple["deepspectrum"])
            egemaps = np.nan_to_num(batch_tuple["egemaps"])
            fasttext = np.nan_to_num(batch_tuple["fasttext"])
            gaze = np.nan_to_num(batch_tuple["gaze"])
            gocar = np.nan_to_num(batch_tuple["gocar"])
            landmarks_2d = np.nan_to_num(batch_tuple["landmarks_2d"])
            landmarks_3d = np.nan_to_num(batch_tuple["landmarks_3d"])
            if self.task_name in ["task1", "task3"]:
                lld = np.nan_to_num(batch_tuple["lld"])
            openpose = np.nan_to_num(batch_tuple["openpose"])
            pdm = np.nan_to_num(batch_tuple["pdm"])
            pose = np.nan_to_num(batch_tuple["pose"])
            vggface = np.nan_to_num(batch_tuple["vggface"])
            xception = np.nan_to_num(batch_tuple["xception"])

            if not ((self.partition == "test") and (not self.are_test_labels_available)):
                arousal = np.nan_to_num(batch_tuple["arousal"])
                valence = np.nan_to_num(batch_tuple["valence"])
                if self.task_name == "task2":
                    topic = np.nan_to_num(batch_tuple["topic"])
                if self.task_name == "task3":
                    trustworthiness = np.nan_to_num(batch_tuple["trustworthiness"])
            support = batch_tuple["support"]

            batch_size_sum += egemaps.shape[0]
            sequence_length = egemaps.shape[1]

            current_batch_size = egemaps.shape[0]

            seq_pos_start = step * self.batch_size
            seq_pos_end = seq_pos_start + egemaps.shape[0]

            # Augment data.
            if self.partition == "train":
                jitter = np.random.normal(scale=self.input_gaussian_noise,
                                          size=au.shape)
                au_plus_jitter = au + jitter

                jitter = np.random.normal(scale=self.input_gaussian_noise,
                                          size=deepspectrum.shape)
                deepspectrum_plus_jitter = deepspectrum + jitter

                jitter = np.random.normal(scale=self.input_gaussian_noise,
                                          size = egemaps.shape)
                egemaps_plus_jitter = egemaps + jitter

                jitter = np.random.normal(scale=self.input_gaussian_noise,
                                          size=fasttext.shape)
                fasttext_plus_jitter = fasttext + jitter

                jitter = np.random.normal(scale=self.input_gaussian_noise,
                                          size=gaze.shape)
                gaze_plus_jitter = gaze + jitter

                jitter = np.random.normal(scale=self.input_gaussian_noise,
                                          size=gocar.shape)
                gocar_plus_jitter = gocar + jitter

                jitter = np.random.normal(scale=self.input_gaussian_noise,
                                          size=landmarks_2d.shape)
                landmarks_2d_plus_jitter = landmarks_2d + jitter

                jitter = np.random.normal(scale=self.input_gaussian_noise,
                                          size=landmarks_3d.shape)
                landmarks_3d_plus_jitter = landmarks_3d + jitter

                if self.task_name in ["task1", "task3"]:
                    jitter = np.random.normal(scale=self.input_gaussian_noise,
                                              size=lld.shape)
                    lld_plus_jitter = lld + jitter

                jitter = np.random.normal(scale=self.input_gaussian_noise,
                                          size=openpose.shape)
                openpose_plus_jitter = openpose + jitter

                jitter = np.random.normal(scale=self.input_gaussian_noise,
                                          size=pdm.shape)
                pdm_plus_jitter = pdm + jitter

                jitter = np.random.normal(scale=self.input_gaussian_noise,
                                          size=pose.shape)
                pose_plus_jitter = pose + jitter

                jitter = np.random.normal(scale=self.input_gaussian_noise,
                                              size=vggface.shape)
                vggface_synced_plus_jitter = vggface + jitter

                jitter = np.random.normal(scale=self.input_gaussian_noise,
                                          size=xception.shape)
                xception_plus_jitter = xception + jitter

            else:
                au_plus_jitter = au
                deepspectrum_plus_jitter = deepspectrum
                egemaps_plus_jitter = egemaps
                fasttext_plus_jitter = fasttext
                gaze_plus_jitter = gaze
                gocar_plus_jitter = gocar
                landmarks_2d_plus_jitter = landmarks_2d
                landmarks_3d_plus_jitter = landmarks_3d
                if self.task_name in ["task1", "task3"]:
                    lld_plus_jitter = lld
                openpose_plus_jitter = openpose
                pdm_plus_jitter = pdm
                pose_plus_jitter = pose
                vggface_synced_plus_jitter = vggface
                xception_plus_jitter = xception

            feed_dict = {k: v for k, v in self.input_feed_dict.items()}
            feed_dict = replace_dict_value(feed_dict, "batch_size", egemaps.shape[0])
            feed_dict = replace_dict_value(feed_dict, "sequence_length", egemaps.shape[1])

            feed_dict = replace_dict_value(feed_dict, "au", au_plus_jitter)
            feed_dict = replace_dict_value(feed_dict, "deepspectrum", deepspectrum_plus_jitter)
            feed_dict = replace_dict_value(feed_dict, "egemaps", egemaps_plus_jitter)
            feed_dict = replace_dict_value(feed_dict, "fasttext", fasttext_plus_jitter)
            feed_dict = replace_dict_value(feed_dict, "gaze", gaze_plus_jitter)
            feed_dict = replace_dict_value(feed_dict, "gocar", gocar_plus_jitter)
            feed_dict = replace_dict_value(feed_dict, "landmarks_2d", landmarks_2d_plus_jitter)
            feed_dict = replace_dict_value(feed_dict, "landmarks_3d", landmarks_3d_plus_jitter)
            if self.task_name in ["task1", "task3"]:
                feed_dict = replace_dict_value(feed_dict, "lld", lld_plus_jitter)
            feed_dict = replace_dict_value(feed_dict, "openpose", openpose_plus_jitter)
            feed_dict = replace_dict_value(feed_dict, "pdm", pdm_plus_jitter)
            feed_dict = replace_dict_value(feed_dict, "pose", pose_plus_jitter)
            feed_dict = replace_dict_value(feed_dict, "vggface", vggface_synced_plus_jitter)
            feed_dict = replace_dict_value(feed_dict, "xception", xception_plus_jitter)

            if not ((self.partition == "test") and (not self.are_test_labels_available)):
                feed_dict = replace_dict_value(feed_dict, "arousal", arousal)
                feed_dict = replace_dict_value(feed_dict, "valence", valence)
                if self.task_name == "task2":
                    feed_dict = replace_dict_value(feed_dict, "topic", topic)
                if self.task_name == "task3":
                    feed_dict = replace_dict_value(feed_dict, "trustworthiness", trustworthiness)
            else:
                feed_dict = {k: v for k, v in feed_dict.items() if v not in ["arousal",
                                                                             "valence",
                                                                             "trustworthiness",
                                                                             "topic"]}
            feed_dict = replace_dict_value(feed_dict, "support", support)

            out_tf = list()
            out_tf.append(self.pred)
            optimizer_index = None
            loss_index = None
            if self.optimizer is not None:
                out_tf.append(self.optimizer)
                optimizer_index = len(out_tf) - 1
            if self.loss is not None:
                out_tf.append(self.loss)
                loss_index = len(out_tf) - 1

            out_np = self.sess.run(out_tf,
                                   feed_dict=feed_dict)

            if not ((self.partition == "test") and (not self.are_test_labels_available)):
                stored_variables["arousal"]["true"][seq_pos_start:seq_pos_end, :] = np.squeeze(arousal)
                stored_variables["valence"]["true"][seq_pos_start:seq_pos_end, :] = np.squeeze(valence)
                if self.task_name == "task2":
                    stored_variables["topic"]["true"][seq_pos_start:seq_pos_end, :] = np.squeeze(topic)
                if self.task_name == "task3":
                    stored_variables["trustworthiness"]["true"][seq_pos_start:seq_pos_end, :] = np.squeeze(trustworthiness)
            stored_variables["support"][seq_pos_start:seq_pos_end, :support.shape[1]] = np.squeeze(support)
            if self.task_name in ["task1", "task3"]:
                stored_variables["arousal"]["pred"][seq_pos_start:seq_pos_end, :] = out_np[0][:, :, 0].reshape(
                    (self.batch_size,
                     self.seq_length))
                stored_variables["valence"]["pred"][seq_pos_start:seq_pos_end, :] = out_np[0][:, :, 1].reshape(
                    (self.batch_size,
                     self.seq_length))
                if self.task_name == "task3":
                    stored_variables["trustworthiness"]["pred"][seq_pos_start:seq_pos_end, :] = out_np[0][:, :,
                                                                                                2].reshape(
                        (self.batch_size,
                         self.seq_length))
            elif self.task_name == "task2":
                stored_variables["arousal"]["pred"][seq_pos_start:seq_pos_end, :] = out_np[0][:, 0:3].reshape(
                    (self.batch_size,
                     3))
                stored_variables["valence"]["pred"][seq_pos_start:seq_pos_end, :] = out_np[0][:, 3:6].reshape(
                    (self.batch_size,
                     3))
                stored_variables["topic"]["pred"][seq_pos_start:seq_pos_end, :] = out_np[0][:,
                                                                                                6:16].reshape(
                    (self.batch_size,
                     10))

            if self.loss is not None:
                stored_variables["loss"] = out_np[loss_index]

        for target in self.targets:
            if not ((self.partition == "test") and (not self.are_test_labels_available)):
                stored_variables[target]["true"] = stored_variables[target]["true"][:batch_size_sum, :]
            stored_variables[target]["pred"] = stored_variables[target]["pred"][:batch_size_sum, :]
            stored_variables[target] = dict_to_struct(stored_variables[target])

        stored_variables["support"] = stored_variables["support"][:batch_size_sum, :]

        stored_variables = dict_to_struct(stored_variables)

        return stored_variables, subject_to_id


def CCC(X1, X2):
    x_mean = np.nanmean(X1)
    y_mean = np.nanmean(X2)
    x_var = 1.0 / (len(X1) - 1) * np.nansum((X1 - x_mean) ** 2)
    y_var = 1.0 / (len(X2) - 1) * np.nansum((X2 - y_mean) ** 2)

    covariance = np.nanmean((X1 - x_mean) * (X2 - y_mean))
    return (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)


def CC(X1, X2):
    ccpea = pd.DataFrame({'X': X1, 'Y': X2})
    return ccpea['X'].corr(ccpea['Y'])


# def flatten_list(l):
#     return list(np.array(l).flat)


def flatten_score(y, y_pred):
    y_flat = y
    y_pred_flat = y_pred

    cc = CC(y_flat, y_pred_flat)  # constant conditional correlation
    ccc = CCC(y_flat, y_pred_flat)  # concordance correlation coefficient
    mae = mean_absolute_error(y_flat, y_pred_flat)  # mean absolute error

    return ccc, cc, mae
