import json
import os
import time
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

import keras
import keras.backend as K
import tensorflow as tf
from keras import metrics


def precision(y_true, y_pred):
    y_true, y_pred = K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) # how many
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    if predicted_positives == 0: return 0
    return TP / predicted_positives

def recall(y_true, y_pred):
    y_true, y_pred = K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    if possible_positives == 0: return 0
    return TP / possible_positives

def f1_score(y_true, y_pred):
    y_true, y_pred = K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    fscore = 2 * (p * r) / (p + r + K.epsilon())
    return fscore


def data_preprocessing(filename):
  lines = open(filename, 'r').read().split('\n')
  lines.reverse()
  prems, hypos, label = [], [], []
  while lines:
    try:
      _prem = lines.pop().split()[1:]
      _hypo = lines.pop().split()[1:]
      _label = lines.pop().split()[0]
    except IndexError:
      break
    prems.append(' '.join(_prem))
    hypos.append(' '.join(_hypo))
    label.append({'Y':1,'N':0}[_label])
  return prems, hypos, label

def save_train_data(filename):
  prems, hypos, label = data_preprocessing(filename)
  assert len(prems) == len(hypos) == len(label)
  spt = int(len(prems) * 0.9)
  open('RTE_train.json', 'w').write(json.dumps([prems[:spt], hypos[:spt], label[:spt]]))
  open('RTE_valid.json', 'w').write(json.dumps([prems[spt:], hypos[spt:], label[spt:]]))
  print('train example:\n', prems[88], '\n', hypos[88], '\n', label[88])

def save_test_data(filename):
  prems, hypos, label = data_preprocessing(filename)
  assert len(prems) == len(hypos) == len(label)
  open('RTE_test.json', 'w').write(json.dumps([prems, hypos, label]))
  print('test example:\n', prems[88], '\n', hypos[88], '\n', label[88])


def merge_data_with_snli():
  strn = json.loads(open('train.json', 'r').read())
  rtrn = json.loads(open('RTE_train.json', 'r').read())

  for i, label in enumerate(strn[2]):
    if label == 1: continue
    if label == 2: label = 1
    rtrn[0].append(strn[0][i])
    rtrn[1].append(strn[1][i])
    rtrn[2].append(label)

  open('RTE_train.json', 'w').write(json.dumps(rtrn))



if __name__ == '__main__':
  # save_train_data('RTE/RTE_train.txt')
  # save_test_data('RTE/RTE_test.txt')
  merge_data_with_snli()
  pass

