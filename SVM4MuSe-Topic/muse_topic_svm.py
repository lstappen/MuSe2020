#!/usr/bin/python3

import numpy as np
import pandas as pd
from sklearn import svm
import metrics as metric
from warnings import filterwarnings
filterwarnings('ignore')

print('\n MuSe2020 Sub-Challenge MuSe-Topic (Emotion)')

feature_set = 'egemaps'
label_options = ['arousal','valence']
partition_info = pd.read_csv('data/processed_tasks/metadata/partition.csv')
classes= [0,1,2]

complexities = [1e-5,1e-4,1e-3,1e-2,1e-1,1e0]  # SVM complexities (linear kernel)

feat_conf = {'egemaps': (88, 1, ',', 'infer'),
             'deepspectrum': (4096, 1, ',', 'infer'),
             'vggface': (512, 1, ',', 'infer'),
             'fasttext': (300, 1, ',', 'infer'),
             'xception': (2048, 1, ',', 'infer')}

num_feat = feat_conf[feature_set][0]
ind_off  = feat_conf[feature_set][1]
sep      = feat_conf[feature_set][2]
header   = feat_conf[feature_set][3]

for label in label_options:
   
    train_lab,train_feat,devel_lab,devel_feat,test_lab,test_feat= [],[],[],[],[],[]

   
    feature_folder 	= 'c2_muse_topic/feature_segments/egemaps_aligned/'
    label_folder 	= 'c2_muse_topic/label_segments/'+ label + '/'
   
    print('\n ' + feature_set + ': ' + label)
   
    print('\n Preparing Partitions')
    for index, row in partition_info.iterrows():
        filename_id = str(row['Id']) + '.csv'
        row_partition = row['Proposal']
        label_df = pd.read_csv(label_folder + filename_id, index_col=None, sep=sep, header=header, dtype=np.float64)
        feature_df = pd.read_csv(feature_folder + feature_set + '/'+filename_id, index_col=None, sep=sep, header=header, usecols=range(ind_off, num_feat + ind_off), dtype=np.float64)
        feature_df = feature_df.groupby(['segment_id']).agg('mean') 
        if row_partition == 'train':
            train_feat.append(feature_df)
            train_lab.append(label_df)
        if row_partition == 'devel':
            devel_feat.append(feature_df)
            devel_lab.append(label_df)
        if row_partition == 'test':
            label_df['id'] = filename_id[:-4]
            label_df['prediction_topic'] = 0 # dummy unused column, for prediction file 
            test_feat.append(feature_df)
            test_lab.append(label_df)

    y_train 	= pd.concat(train_lab, axis=0).reset_index()
    y_train 	= y_train['class_id']
    X_train		= pd.concat(train_feat, axis=0).reset_index()
   
    y_devel 	= pd.concat(devel_lab, axis=0).reset_index()
    y_devel 	= y_devel['class_id']
    X_devel 	= pd.concat(devel_feat, axis=0).reset_index()
   
    y_test 	= pd.concat(test_lab, axis=0).reset_index()
    y_test 	= y_test[['id','segment_id','prediction_topic']]
    X_test 	= pd.concat(test_feat, axis=0).reset_index()
    
    y_traindevel = np.concatenate((y_train, y_devel))
    X_traindevel = np.concatenate((X_train, X_devel))

    print('\n Begin training SVM... (may take a while)')
    uar_scores,fone_scores = [],[]
    for comp in complexities:
        print('\nComplexity {0:.6f}'.format(comp))
        clf = svm.LinearSVC(C=comp, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_devel)
        uar_scores.append(metric.uar(y_devel, y_pred))
        fone_scores.append(metric.f1(y_devel, y_pred))

        print('UAR on Devel {0:.3f}'.format(uar_scores[-1]))
        print('F1 on Devel {0:.3f}'.format(fone_scores[-1]))

    optimum_complexity = complexities[np.argmax(uar_scores)]
    uar = np.max(uar_scores)
    f1 	=  np.max(fone_scores)
   
    print('\nOptimum complexity: {0:.6f}, maximum UAR: {1:.3f}, F1: {2:.3f}'.format(optimum_complexity, uar, f1))
    print('Devel Combined Score: {0:.3f} '.format(metric.combined_task2(f1,uar)))

    print('\nCalculating Test Predictions')
    clf = svm.LinearSVC(C=optimum_complexity, random_state=0)
    clf.fit(X_traindevel, y_traindevel)
    if label == 'arousal':
        y_pred_arousal = clf.predict(X_test)
    if label == 'valence':
        y_pred_valence = clf.predict(X_test)

pred_file_name =  feature_set +'_test.csv'
print('Writing file ' + pred_file_name + '\n')
prediction_df = pd.DataFrame(data={'id': y_test['id'],
                        'segment_id': y_test['segment_id'].astype(int),
                        'prediction_arousal': y_pred_arousal.astype(int),
                        'prediction_valence': y_pred_valence.astype(int),
                        'prediction_topic': y_test['prediction_topic'].astype(int),},columns=['id','segment_id', 'prediction_arousal','prediction_valence', 'prediction_topic'])
prediction_df.to_csv(pred_file_name, index=False)
