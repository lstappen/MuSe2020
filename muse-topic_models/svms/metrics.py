import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, f1_score, recall_score


def flatten_list(l):
    return np.array(list(np.array(l).flat))


# Regression
## We calculate the metrics over all concatenated segments. 
## Hence, we avoid taking different lengths of the segments into account.
def combined_task1(arousal_CCC, valence_CCC):
    return round((0.5 * arousal_CCC + 0.5 * valence_CCC), 4)


def CCC(X1, X2):
    x_mean = np.nanmean(X1)
    y_mean = np.nanmean(X2)
    x_var = 1.0 / (len(X1) - 1) * np.nansum((X1 - x_mean) ** 2)
    y_var = 1.0 / (len(X2) - 1) * np.nansum((X2 - y_mean) ** 2)

    covariance = np.nanmean((X1 - x_mean) * (X2 - y_mean))
    return round((2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2), 4)


def CC(X1, X2):
    ccpea = pd.DataFrame({'X': X1, 'Y': X2})
    return round(ccpea['X'].corr(ccpea['Y']), 4)


def flatten_score_task13(y, y_pred):
    y_flat = flatten_list(y)
    y_pred_flat = flatten_list(y_pred)

    if isinstance(y_flat[0], np.ndarray):
        y_flat = np.concatenate(y_flat)
        y_pred_flat = np.concatenate(y_pred_flat)

    cc = CC(y_flat, y_pred_flat)  # constant conditional correlation
    ccc = CCC(y_flat, y_pred_flat)  # concordance correlation coefficient
    mae = round(mean_absolute_error(y_flat, y_pred_flat), 4)  # mean absolute error

    return ccc, cc, mae


# Classification
# custom metrics
def f1(y_true, y_pred):
    return np.round(f1_score(y_true, y_pred, average='micro') * 100, 3)


def uar(y_true, y_pred):
    return round(recall_score(y_true, y_pred, average='macro') * 100, 3)


def combined_task2(f1, uar):
    return round((0.66 * f1 + 0.34 * uar), 3)


def flatten_score_task2(y, y_pred):
    y_flat = flatten_list(y)
    y_pred_flat = flatten_list(y_pred)

    if isinstance(y_flat[0], np.ndarray):
        y_flat = np.concatenate(y_flat)
        y_pred_flat = np.concatenate(y_pred_flat)

    f1_score = f1(y_flat, y_pred_flat)
    uar_score = uar(y_flat, y_pred_flat)
    combined = combined_task2(f1_score, uar_score)

    return f1_score, uar_score, combined


def read_agg_label_files(filename):
    df = pd.read_csv(filename, delimiter=',')
    return df


def score_partition(task, partition, y_pred_df, processed_path='../data/processed_tasks/'):
    label_path = os.path.join(processed_path, task, 'label_segments', 'aggregated')

    task_columns = {'task1': ['id', 'timestamp', 'prediction_arousal', 'prediction_valence'],
                    'task2': ['id', 'segment_id', 'prediction_arousal', 'prediction_valence', 'prediction_topic'],
                    'task3': ['id', 'timestamp', 'prediction_trustworthiness']}

    if sorted(list(y_pred_df.columns)) != sorted(task_columns[task]):
        print("Task {} needs {} columns but has {}.".format(task, sorted(task_columns[task]),
                                                            sorted(list(y_pred_df.columns))))
        exit()

    label_file = read_agg_label_files(os.path.join(label_path, partition + '.csv'))
    y_pred_df = y_pred_df[task_columns[task]].set_index(task_columns[task][:2])
    label_file = label_file.set_index(task_columns[task][:2])

    combined = pd.concat([label_file, y_pred_df], axis=1, join='outer').reset_index()
    if combined.isnull().values.any():
        print(
            "Join of labels and prediction results in NaN values. Please check if there is a prediction for each timestamp.")
        print(combined[combined.isna().any(axis=1)])
        exit()

    if task == 'task1':
        arousal_CCC = flatten_score_task13(combined.label_arousal.values, combined.prediction_arousal.values)[0]
        valence_CCC = flatten_score_task13(combined.label_valence.values, combined.prediction_valence.values)[0]
        combined_CCC = combined_task1(arousal_CCC, valence_CCC)

        print("{} ({}) arousal CCC: {}, valence CCC: {}, combined: {}".format(task, partition, arousal_CCC, valence_CCC,
                                                                              combined_CCC))

    elif task == 'task2':
        arousal_f1_score, arousal_uar_score, arousal_combined = flatten_score_task2(combined.label_arousal.values,
                                                                                    combined.prediction_arousal.values)
        valence_f1_score, valence_uar_score, valence_combined = flatten_score_task2(combined.label_valence.values,
                                                                                    combined.prediction_valence.values)
        topic_f1_score, topic_uar_score, topic_combined = flatten_score_task2(combined.label_topic.values,
                                                                              combined.prediction_topic.values)

        print("{} ({}) arousal F1: {}, arousal UAR: {}, combined: {}".format(task, partition, arousal_f1_score,
                                                                             arousal_uar_score, arousal_combined))
        print("{} ({}) arousal F1: {}, valence UAR: {}, combined: {}".format(task, partition, valence_f1_score,
                                                                             valence_uar_score, valence_combined))
        print("{} ({}) topic F1: {}, topic UAR: {}, topic: {}".format(task, partition, topic_f1_score, topic_uar_score,
                                                                      topic_combined))

    elif task == 'task3':
        trustworthiness_CCC = \
        flatten_score_task13(combined.label_trustworthiness.values, combined.prediction_trustworthiness.values)[0]

        print("{} ({}) trustworthiness CCC: {}".format(task, partition, trustworthiness_CCC))

    else:
        print("specify task1, task2 or task3. exit")
        exit()


if __name__ == '__main__':

    # pandas/ file based
    for task in ['task1', 'task2', 'task3']:
        for partition in ['train', 'devel', 'test']:
            df = pd.read_csv(os.path.join('fake_predictions', task, partition + '.csv'), delimiter=',')
            score_partition(task, partition, y_pred_df=df)

 



