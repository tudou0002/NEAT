import argparse

import pandas as pd
import numpy as np

import ast
import copy

def apply_lit(input):
  if input == 'set()':
    return set()
  else:
    return set(ast.literal_eval(input))

def Containment_IoU(input1, input2):
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

Containment_IoU = np.vectorize(Containment_IoU)


def Exact_Set(input1, input2):

  input1 = set(input1)
  input2 = set(input2)

  if input1 == input2:
    return 1
  else:
    return 0

Exact_Set = np.vectorize(Exact_Set)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='to do')
    parser.add_argument('-p', type=str, default='./data/Crowd_Majority_and_others.tsv', help='prediction filepath')
    parser.add_argument('-pc', type=str, default='result', help='prediction column name')
    parser.add_argument('-t', type=str, default='./data/Crowd_Majority_and_others.tsv', help='ground truth filepath')
    parser.add_argument('-tc', type=str, default='Crowd_Majority', help='ground truth column name')
    parser.add_argument('--mode', type=str, default='Containment_IoU', help='how to do comparison. Choices: Containment_IoU, Exact_Set')

    args = parser.parse_args()

    if args.mode not in ['Containment_IoU','Exact_Set']:
      print('Mode not recognized')
      exit()

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

    else: # no id, so assume entries are aligned
      pred_col = pred_df[args.pc]
      true_col = true_df[args.tc]

      # only compare where we have a "ground truth"
      pred_col = pred_col[true_col != '']
      pred_col = pred_col[~true_col.isna()]
      true_col = true_col[true_col != '']
      true_col = true_col[~true_col.isna()]

      # convert strings to lists
      pred_col = pred_col.apply(apply_lit)
      true_col = true_col.apply(apply_lit)

    if args.mode == 'Containment_IoU':
      comparison = Containment_IoU(pred_col, true_col)
      print(np.mean(comparison))
    elif args.mode == 'Exact_Set':
      comparison = Exact_Set(pred_col, true_col)
      print(np.mean(comparison))