from pprint import pprint
from functools import reduce
import json
import os
import re
import tarfile
import tempfile

import numpy as np
np.random.seed(1337)  # for reproducibility

'''
300D Model - Train / Test (epochs)
=-=-=
Batch size = 512
Fixed GloVe
- 300D SumRNN + Translate + 3 MLP (1.2 million parameters) - 0.8315 / 0.8235 / 0.8249 (22 epochs)
- 300D GRU + Translate + 3 MLP (1.7 million parameters) - 0.8431 / 0.8303 / 0.8233 (17 epochs)
- 300D LSTM + Translate + 3 MLP (1.9 million parameters) - 0.8551 / 0.8286 / 0.8229 (23 epochs)

Following Liu et al. 2016, I don't update the GloVe embeddings during training.
Unlike Liu et al. 2016, I don't initialize out of vocabulary embeddings randomly and instead leave them zeroed.

The jokingly named SumRNN (summation of word embeddings) is 10-11x faster than the GRU or LSTM.

Original numbers for sum / LSTM from Bowman et al. '15 and Bowman et al. '16
=-=-=
100D Sum + GloVe - 0.793 / 0.753
100D LSTM + GloVe - 0.848 / 0.776
300D LSTM + GloVe - 0.839 / 0.806
'''

import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils

LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

def read_data():
  trn = json.loads(open('train.json', 'r').read())
  vld = json.loads(open('validation.json', 'r').read())
  tst = json.loads(open('test.json', 'r').read())

  trn[2] = np_utils.to_categorical(trn[2], len(LABELS))
  vld[2] = np_utils.to_categorical(vld[2], len(LABELS))
  tst[2] = np_utils.to_categorical(tst[2], len(LABELS))
  return trn,vld,tst

print('Reading Data...',end='')
training,validation,test = read_data()
print('Succucess!,example=',training[0][0],training[1][0])

# word2vec ?
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])

# Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
# Summation of word embeddings
VOCAB = len(tokenizer.word_counts) + 1
#RNN = recurrent.LSTM
#RNN = lambda *args, **kwargs: Bidirectional(recurrent.LSTM(*args, **kwargs))
RNN = recurrent.GRU
#RNN = lambda *args, **kwargs: Bidirectional(recurrent.GRU(*args, **kwargs))
LAYERS = 1
USE_GLOVE = True
TRAIN_EMBED = False
EMBED_HIDDEN_SIZE = 300 # Must Follow GloVe Demision Size!!!
SENT_HIDDEN_SIZE = 300
BATCH_SIZE = 512
PATIENCE = 4 # 8
MAX_EPOCHS = 42
MAX_LEN = 42
DP = 0.2
L2 = 4e-6
ACTIVATION = 'relu'
OPTIMIZER = 'rmsprop'
print('RNN / Embed / Sent = {}, {}, {}'.format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE))
print('GloVe / Trainable Word Embeddings = {}, {}'.format(USE_GLOVE, TRAIN_EMBED))

# Q1: pad in FRONT of the sent
to_index = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
prepare_data = lambda data: (to_index(data[0]), to_index(data[1]), data[2])

training = prepare_data(training)
validation = prepare_data(validation)
test = prepare_data(test)

print('Indexize Finished, example = ',training[0][0],training[1][0])

print('Build model...')
print('Vocab size =', VOCAB)

GLOVE_STORE = 'precomputed_glove.weights'
if USE_GLOVE:
  if not os.path.exists(GLOVE_STORE + '.npy'): # 是否已有处理好的数据
    print('Computing GloVe')

    embeddings_index = {}
    f = open('/my_data/glove.840B.300d.txt')
    for line in f:
      values = line.split(' ')
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs # 300D
    f.close()

    # prepare embedding matrix
    embedding_matrix = np.zeros((VOCAB, EMBED_HIDDEN_SIZE)) # V * 300D
    for word, i in tokenizer.word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
      else: #  matrix[i] would not change, which is zeros
        print('Missing from GloVe: {}'.format(word))

    np.save(GLOVE_STORE, embedding_matrix)

  print('Loading GloVe')
  embedding_matrix = np.load(GLOVE_STORE + '.npy')

  print('Total number of null word embeddings:')
  print(np.sum(np.sum(embedding_matrix, axis=1) == 0))

  embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], # Qs:weights? No Official Doc!
                    input_length=MAX_LEN, trainable=TRAIN_EMBED)
else:
  embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, input_length=MAX_LEN)


#
#
#         Keras Starts Here !!!
#
#

rnn_kwargs = dict(output_dim=SENT_HIDDEN_SIZE, dropout_W=DP, dropout_U=DP)
# Qs: What's SumEmbeddings?????
SumEmbeddings = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1), output_shape=(SENT_HIDDEN_SIZE, ))

translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

premise = Input(shape=(MAX_LEN,), dtype='int32')
hypothesis = Input(shape=(MAX_LEN,), dtype='int32')

prem = embed(premise)
hypo = embed(hypothesis)

prem = translate(prem)
hypo = translate(hypo)

if RNN and LAYERS > 1:
  for l in range(LAYERS - 1):
    rnn = RNN(return_sequences=True, **rnn_kwargs)
    prem = BatchNormalization()(rnn(prem))
    hypo = BatchNormalization()(rnn(hypo))
rnn = SumEmbeddings if not RNN else RNN(return_sequences=False, **rnn_kwargs)
prem = rnn(prem)
hypo = rnn(hypo)
prem = BatchNormalization()(prem)
hypo = BatchNormalization()(hypo)

joint = merge([prem, hypo], mode='concat')
joint = Dropout(DP)(joint)
for i in range(3): # 3 layer MLP ?
  joint = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, W_regularizer=l2(L2) if L2 else None)(joint)
  joint = Dropout(DP)(joint)
  joint = BatchNormalization()(joint)

pred = Dense(len(LABELS), activation='softmax')(joint)

model = Model(input=[premise, hypothesis], output=pred)
model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary() # print a summary

print('Training')
#_, tmpfn = tempfile.mkstemp()
tmpfn = 'checkpoint'
# Save the best model during validation and bail out of training early if we're not improving
callbacks = [EarlyStopping(patience=PATIENCE),
             ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)]
model.fit([training[0], training[1]], training[2],
          batch_size=BATCH_SIZE,
          nb_epoch=MAX_EPOCHS,
          validation_data=([validation[0], validation[1]], validation[2]), callbacks=callbacks)

# Restore the best found model during validation
model.load_weights(tmpfn)

loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

