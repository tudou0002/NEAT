import argparse

import pandas as pd
import numpy as np

import ast
import copy

def apply_lit(input):
  if input == 'set()':
    return set()
  else:
    try:
      return set(ast.literal_eval(input))
    except:
      # TJBatch extractor results need to be split using the ; delimiter
      if pd.isna(input): # THESE ARE THE LINES ADDED
        return [] # THESE ARE THE LINES ADDED
      return set(input.split(';'))

def apply_lower(input):
  return set([x.lower() for x in list(input)])

def Containment_IoU(input1, input2): # pred, true
  intersect_count = 0
  union_count = 0

  input1 = list(input1)
  input2 = list(input2)

  for i in input1:
    remain1 = copy.copy(input1)
    remain1.remove(i)
    for j in remain1:
      if i in j:
        try: 
          input1.remove(i)
        except:
          continue
  for i in input2:
    remain2 = copy.copy(input2)
    remain2.remove(i)
    for j in remain2:
      if i in j:
        try: 
          input2.remove(i)
        except:
          continue

  for i in input1:
    union_count += 1
    if ~(' ' in i):
      for j in input2:
        if (i in j) or (j in i):
          intersect_count += 1
          input2.remove(j)
    else:
      for s in i.split(' '):
        for j in input2:
          if (i in j) or (j in i):
            intersect_count += 1
            input2.remove(j)
    
  union_count += len(input2)
  
  mod_IoU = intersect_count / union_count if union_count > 0 else 1

  return mod_IoU

def Exact_Set(input1, input2): # pred, true

  input1 = set(input1)
  input2 = set(input2)

  if input1 == input2:
    return 1
  else:
    return 0

def Exact_F1(pred, true):
  pred = set(pred)
  true = set(true)
  
  tp = 0
  fp = 0
  fn = 0

  for i in pred:
    if i in true:
      tp += 1
    else:
      fp += 1
  for i in true:
    if i not in pred:
      fn += 1

  return tp, fp, fn

Containment_IoU = np.vectorize(Containment_IoU)
Exact_Set = np.vectorize(Exact_Set)
Exact_F1 = np.vectorize(Exact_F1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='to do')
    parser.add_argument('-p', type=str, default='./data/Crowd_Majority_and_others.tsv', help='prediction filepath')
    parser.add_argument('-pc', type=str, default='result', help='prediction column name')
    parser.add_argument('-t', type=str, default='./data/Crowd_Majority_and_others.tsv', help='ground truth filepath')
    parser.add_argument('-tc', type=str, default='Crowd_Majority', help='ground truth column name')

    args = parser.parse_args()


    pred_df = pd.read_csv(args.p, sep='\t')
    true_df = pd.read_csv(args.t, sep='\t')

    if 'id' in pred_df.columns and 'id' in true_df.columns:
      # take columns of interest and merge on id
      pred_df = pred_df[['id',args.pc]]
      true_df = true_df[['id',args.tc]]
      merged_df = pred_df.merge(true_df, on='id')

      merged_df = merged_df[merged_df[args.tc] != ''] # only compare where we have a "ground truth"
      merged_df = merged_df[~merged_df[args.tc].isna()]

      # convert strings to lists
      merged_df[args.pc] = merged_df[args.pc].apply(apply_lit)
      merged_df[args.tc] = merged_df[args.tc].apply(apply_lit)

      pred_col = merged_df[args.pc]
      true_col = merged_df[args.tc]
      
      # lowercase
      pred_col = merged_df[pc].apply(apply_lower)
      true_col = merged_df[tc].apply(apply_lower)

    else: # no id, so assume entries are aligned
      pred_col = pred_df[args.pc]
      true_col = true_df[args.tc]

      # only compare where we have a "ground truth"
      pred_col = pred_col[true_col != '']
      pred_col = pred_col[~true_col.isna()]
      true_col = true_col[true_col != '']
      true_col = true_col[~true_col.isna()]

      # convert strings to lists and lowercase
      pred_col = pred_col.apply(apply_lit).apply(apply_lower)
      true_col = true_col.apply(apply_lit).apply(apply_lower)

    
    comparison = Containment_IoU(pred_col, true_col)
    print('Containment IoU:', np.mean(comparison))
    comparison = Exact_Set(pred_col, true_col)
    print('Full set exact match accuracy:', np.mean(comparison))


    comparison = Exact_F1(pred_col, true_col)
    tp = np.sum(comparison[0])
    fp = np.sum(comparison[1])
    fn = np.sum(comparison[2])
    
    avg_precision = tp / (tp + fp)
    avg_recall = tp / (tp + fn)
    avg_f1 =  2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

    print('Individual exact match precision:', np.mean(avg_precision))
    print('Individual exact match recall:', np.mean(avg_recall))
    print('Individual exact match F1:', np.mean(avg_f1))
