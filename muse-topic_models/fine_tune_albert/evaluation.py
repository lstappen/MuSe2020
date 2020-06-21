import numpy as np
import pandas as pd
import pickle
import json
import os
import matplotlib
matplotlib.use('AGG')
import csv
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import mean_absolute_error, f1_score, recall_score, accuracy_score

# custom metrics
def f1(y_true, y_pred):
    return round(f1_score(y_true, y_pred, average = 'micro')*100, 3) 

def uar(y_true, y_pred):
    return round(recall_score(y_true, y_pred, average = 'macro')*100, 3) 

def flatten_list(l):
    return list(np.array(l).flat)

def flatten_score_task2(y, y_pred):

    y_flat = flatten_list(y)
    y_pred_flat = flatten_list(y_pred)
    rho, pval = stats.spearmanr(y_flat, y_pred_flat)
    mae = mean_absolute_error(y_flat, y_pred_flat) 

    return rho, pval, mae

# sklearn cf to visualise the class predictions
def plot_confusion_matrix(y_true, y_pred, 
                          normalize = False, 
                          title = None, 
                          ticklabels = None
                          , path = None
                          , cmap = plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize = (10, 10))
    im = ax.imshow(cm, interpolation = 'nearest', cmap = cmap)
    ax.figure.colorbar(im, ax = ax)

    if not ticklabels:
        ticklabels = classes
    ax.set(xticks = np.arange(cm.shape[1]), 
           yticks = np.arange(cm.shape[0]), 
           # ... and label them with the respective list entries
           xticklabels = ticklabels, yticklabels = ticklabels, 
           title = title, ylabel = 'Truth', xlabel = 'Prediction')

    plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", 
             rotation_mode = "anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), 
                    ha = "center", va = "center", 
                    color = "white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    if path:
        fig.savefig(os.path.join(path, title+'.png'))
    return ax


def export_ys(Param, y, y_pred, partition_name):
    y_path = os.path.join(Param['output_dir'], 'y')
    y_pred_path = os.path.join(Param['output_dir'], 'y_pred')

    if not os.path.exists(y_path):
        os.makedirs(y_path)
    if not os.path.exists(y_pred_path):
        os.makedirs(y_pred_path)

    with open(os.path.join(y_pred_path, "{}.pkl".format(partition_name)), "wb" ) as f:
        pickle.dump(y_pred,  f)
    with open(os.path.join(y_path, "{}.pkl".format(partition_name)), "wb" ) as f:
        pickle.dump(y, f)


def classification_results(Param, model, df, y, y_names, partition_name, class_name):

    result, model_outputs, wrong_predictions = model.eval_model(df, f1 = f1, uar = uar, acc = accuracy_score)
    print("{}: {}".format(partition_name, result))
    # save pre. calc. results
    with open(Param['output_dir']+'/results_{}.json'.format(partition_name), 'w') as fp:
        json.dump(result, fp)

    y_pred = np.argmax(model_outputs, axis = 1).tolist()
    report = classification_report(list(y), y_pred, target_names = y_names)

    print('classification Report for {} set\n {}'.format(partition_name, report))
    report_df = pd.DataFrame(classification_report(list(y), list(y_pred)
                            , target_names = y_names
                            , output_dict = True
                            , digits = 3)).transpose().round({'support':0})
    report_df['support'] = report_df['support'].apply(int)
    report_df.to_excel(os.path.join(Param['output_dir'], '{}_classification_report.xlsx'.format(partition_name)))
    print('classification report for {} exported.'.format(partition_name))

    plot_confusion_matrix(list(y), list(y_pred), normalize = False, ticklabels = y_names, 
                          title = 'Confusion matrix {}'.format(partition_name), path = Param['output_dir'])
    print('confusion matrix for {} exported.'.format(partition_name))

    results = {}
    results['f1'] = f1(y, y_pred)
    results['uar'] = uar(y, y_pred)
    results['topic_score'] = (0.66 * results['f1'] + 0.34 * results['uar'])
    # if we have an order in the labels
    if class_name in ['arousal', 'valence']:
        results['rho'], results['pval'], results['mae'] = flatten_score_task2(y, y_pred)
        print('{} rho: {} mae: {}'.format(partition_name, results['rho'], results['mae']))

    with open(os.path.join(Param['output_dir'], partition_name+'.csv'), 'w+', newline = "") as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in results.items():
            writer.writerow([key, value])

    print("  ", partition_name)
    for k, v in results.items():
        print("  - {}: {}".format(k,v))

    export_ys(Param, y, y_pred, partition_name)


def regression_results(Param, model, df, y, y_names, partition_name):

    result, y_pred, wrong_predictions = model.eval_model(df, mae = mean_absolute_error)
    print("{}: {}".format(partition_name, result))

    results = {}
    results['rho'], results['pval'], results['mae'] = flatten_score_task2(y, y_pred)
    with open(os.path.join(Param['output_dir'], partition_name+'.csv'), 'w+', newline = "") as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in results.items():
            writer.writerow([key, value])

    # Report results
    with open(Param['output_dir']+'/eval_results_{}.json'.format(partition_name), 'w') as fp:
        json.dump(result, fp)
    export_ys(Param, y, y_pred, partition_name)
