import torch
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, recall_score
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import csv
import pickle
from scipy import stats

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label==1) & (predicted_label==1)))
    tn = float(np.sum((true_label==0) & (predicted_label==0)))
    p = float(np.sum(true_label==1))
    n = float(np.sum(true_label==0))

    return (tp * (n/p) +tn) / (2*n)

# custom metrics
def f1(y_true, y_pred):
    # take label imbalance into account
    return round(f1_score(y_true, y_pred,average='micro')*100,3) 

def uar(y_true, y_pred):
    # unweighted mean
    return round(recall_score(y_true, y_pred,average='macro')*100,3) 

def flatten_list(l):
    return list(np.array(l).flat)

def flatten_score_task2(y, y_pred):

    y_flat = flatten_list(y)
    y_pred_flat = flatten_list(y_pred)

    rho, pval = stats.spearmanr(y_flat,y_pred_flat)
    mae = mean_absolute_error(y_flat, y_pred_flat)  # mean absolute error

    return rho, pval, mae

#analyse the test result
def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          ticklabels=None
                          ,path = None
                          ,cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    if not ticklabels:
        ticklabels = classes
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=ticklabels, yticklabels=ticklabels,
           title=title,
           ylabel='Truth',
           xlabel='Prediction')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    if path:
        fig.savefig(os.path.join(path,title+'.png'))
    return ax

def export_ys(y, y_pred, partition_name, output_dir):
    y_path = os.path.join(output_dir,'y')
    y_pred_path = os.path.join(output_dir,'y_pred')

    if not os.path.exists(y_path):
        os.makedirs(y_path)
    if not os.path.exists(y_pred_path):
        os.makedirs(y_pred_path)

    with open(os.path.join(y_pred_path,"{}.pkl".format(partition_name)), "wb" ) as f:
        pickle.dump(y_pred,  f)
    with open(os.path.join(y_path,"{}.pkl".format(partition_name)), "wb" ) as f:
        pickle.dump(y, f)


def eval_reg_muse(res, truths, partition_name, output_dir,export=False):
    y_pred = res.view(-1).cpu().detach().numpy()
    y = truths.view(-1).cpu().detach().numpy()

    results = {}
    results['rho'], results['pval'], results['mae'] = flatten_score_task2(y, y_pred)

    if export:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir,partition_name+'.csv'), 'w+', newline="") as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in results.items():
                writer.writerow([key, value])
        # Report results
        export_ys(y,y_pred,partition_name,output_dir)
    else:
        print("  ", partition_name)
        print("  - rho: ", results['rho'])
        print("  - mae: ", results['mae'])

        return results['rho']


def eval_class_muse(res, truths, partition_name, output_dir, y_names, export = False):

    y_pred = res.cpu().data.numpy().argmax(axis=1) #y_pred = results.view(-1).cpu().detach().numpy()
    y = truths.cpu().data.numpy() # test_truth = truths.view(-1).cpu().detach().numpy()

    results = {}
    results['f1'] = f1(y, y_pred)
    results['uar'] = uar(y, y_pred)
    results['topic_score'] = round((0.66 * results['f1'] + 0.34 * results['uar']),3)

    report = classification_report(y,y_pred, target_names=y_names, digits = 3)

    if export:
        report_df = pd.DataFrame(classification_report(list(y), list(y_pred)
                                            , target_names=y_names
                                            , output_dict=True, digits = 3)).transpose().round({'support':0})
        report_df['support'] = report_df['support'].apply(int)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        report_df.to_excel(os.path.join(output_dir,'{}_classification_report.xlsx'.format(partition_name)))

        print('classification report for {} exported.'.format(partition_name))
        plot_confusion_matrix(list(y),list(y_pred), normalize=False, ticklabels=y_names,
                              title='Confusion matrix {}'.format(partition_name),path=output_dir)
        print('confusion matrix for {} exported.'.format(partition_name))

        with open(os.path.join(output_dir,partition_name+'.csv'), 'w+', newline="") as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in results.items():
                writer.writerow([key, value])

        print("  ", partition_name)
        print("  - f1: ", results['f1'])
        print("  - uar: ", results['uar'])
        print("  - combined:",results['topic_score'])
        export_ys(y,y_pred,partition_name,output_dir)
    else:
        print("  ", partition_name)
        print("  - f1: ", results['f1'])
        print("  - uar: ", results['uar'])
        print("  - combined:",results['topic_score'])

        print('Classification report for {} set\n {}'.format(partition_name, report))

        return results['topic_score']


