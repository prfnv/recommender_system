import pandas as pd
import numpy as np


def recall_at_k(test, k):
    pred = (
        test
        .sort_values(['clientid', 'itemid', 'predict_proba'], ascending=[True, True, False])
        .groupby(['clientid', 'itemid'])['jointitemid']
        .agg(list)
        .reset_index()
    )
    
    true = (
        test[test['label'] == 1]
        .groupby(['clientid', 'itemid'])['jointitemid']
        .agg(list)
        .reset_index()
    )
    
    pred_true_df = pred.merge(true, on=['clientid', 'itemid'], how='inner', suffixes=('_pred', '_true'))
    
    hits = 0
    alls = 0
    for pred, true in pred_true_df[['jointitemid_pred', 'jointitemid_true']].values:
        for idx in set(pred[:k]):
            if idx in set(true):
                hits += 1
        alls += len(set(true))

    return (hits * 1.) / alls


def map_at_k(test, k):   
    pred = (
        test
        .sort_values(['clientid', 'itemid', 'predict_proba'], ascending=[True, True, False])
        .groupby(['clientid', 'itemid'])['jointitemid']
        .agg(list)
        .reset_index()
    )
    
    true = (
        test[test['label'] == 1]
        .groupby(['clientid', 'itemid'])['jointitemid']
        .agg(list)
        .reset_index()
    )
    
    pred_true_df = pred.merge(true, on=['clientid', 'itemid'], how='inner', suffixes=('_pred', '_true'))
    
    score = 0
    hits=0
    general_score=[]

    for pred, true in pred_true_df[['jointitemid_pred', 'jointitemid_true']].values:
        for i,dx in enumerate(set(pred[:k])):
            if dx in set(true):
                hits += 1
                score += hits/(i+1)
        general_score.append(score/k)
        score=0
        hits=0
        
    return np.mean(general_score)