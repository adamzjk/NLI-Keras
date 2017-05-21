import json
import os
import time
import re
import numpy as np
import tensorflow as tf
from collections import defaultdict

import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils
# import h5py #: requires h5py to store parameters !


def time_count(fn):
  # Funtion wrapper used to memsure time consumption
  def _wrapper(*args, **kwargs):
    start = time.clock()
    fn(*args, **kwargs)
    print("[time_count]: %s cost %fs" % (fn.__name__, time.clock() - start))
  return _wrapper

class NLI_USE_RNN:

  def __init__(self, rnn_type = 'biGRU'):
    # 1, Set Basic Model Parameters
    self.Layers = 1
    self.HiddenSize = 300
    self.BatchSize = 512
    self.Patience = 8
    self.MaxEpoch = 64
    self.MaxLen = 42
    self.DropProb = 0.25
    self.L2Strength = 4e-6
    self.Activate = 'relu'
    self.Optimizer = 'rmsprop'
    self.rnn_type = rnn_type

    # 2, Define Class Variables
    self.Vocab = 0
    self.model = None
    self.GloVe = defaultdict(np.array)
    self.indexer,self.Embed = None, None
    self.train, self.validation, self.test = [],[],[]
    self.Labels = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    self.rLabels = {0:'contradiction', 1:'neutral', 2:'entailment'}

  @staticmethod
  def load_data():
    trn = json.loads(open('train.json', 'r').read())
    vld = json.loads(open('validation.json', 'r').read())
    tst = json.loads(open('test.json', 'r').read())

    trn[2] = np_utils.to_categorical(trn[2], 3)
    vld[2] = np_utils.to_categorical(vld[2], 3)
    tst[2] = np_utils.to_categorical(tst[2], 3)
    return trn, vld, tst

  @time_count
  def prep_data(self,fn=('train.json','validation.json','test.json')):
    # 1, Read raw Training,Validation and Test data
    self.train,self.validation,self.test = self.load_data()

    # 2, Prep Word Indexer: assign each word a number
    self.indexer = Tokenizer(lower=False, filters='')
    self.indexer.fit_on_texts(self.train[0] + self.train[1])
    self.Vocab = len(self.indexer.word_counts) + 1

    # 3, Convert each word in sent to num and zero pad
    prep_alfa = lambda X: pad_sequences(sequences=self.indexer.texts_to_sequences(X),
                                        maxlen=self.MaxLen)
    prep_beta = lambda D: (prep_alfa(D[0]), prep_alfa(D[1]), D[2])

    self.train = prep_beta(self.train)
    self.validation = prep_beta(self.validation)
    self.test = prep_beta(self.test)

  def load_GloVe(self):
    # Creat a embedding matrix for word2vec(use GloVe)
    embed_index = {}
    for line in open('glove.840B.300d.txt','r'):
      value = line.split(' ') # Warning: Can't use split()! I don't know why...
      word = value[0]
      embed_index[word] = np.asarray(value[1:],dtype='float32')
    embed_matrix = np.zeros((self.Vocab,self.HiddenSize))
    unregistered = []
    for word,i in self.indexer.word_index.items():
      vec = embed_index.get(word)
      if vec is None: unregistered.append(word)
      else: embed_matrix[i] = vec
    np.save('GloVe.npy',embed_matrix)
    open('unregisterd_word.txt','w').write(str(unregistered))

  @time_count
  def load_GloVe_dict(self):
    for line in open('glove.840B.300d.txt','r'):
      value = line.split(' ') # Warning: Can't use split()! I don't know why...
      word = value[0]
      self.GloVe[word] = np.asarray(value[1:],dtype='float32')


  @time_count
  def prep_embd(self):
    # Add a Embed Layer to convert word index to vector
    if not os.path.exists('GloVe.npy'):
      self.load_GloVe()
    embed_matrix = np.load('GloVe.npy')
    self.Embed = Embedding(input_dim = self.Vocab,
                           output_dim = self.HiddenSize,
                           input_length = self.MaxLen,
                           trainable = False,
                           weights = [embed_matrix])

  @time_count
  def creat_sumRNN_model(self):
    assert self.rnn_type == 'sumRNN'
    # 1, Sentences Input Layer
    premise = Input(shape=(self.MaxLen,),dtype='int32')
    hypothesis = Input(shape=(self.MaxLen,),dtype='int32')
    FullyConnected = TimeDistributed(Dense(self.HiddenSize, activation=self.Activate))
    P,H = FullyConnected(self.Embed(premise)),FullyConnected(self.Embed(hypothesis))

    # 2, A Fast & Effective Encoder: Sum Embedding Layer
    SumEmbeddings = keras.layers.core.Lambda(lambda X:
                                             K.sum(X, axis=1),
                                             output_shape=(self.HiddenSize,))
    P,H = SumEmbeddings(P),SumEmbeddings(H)
    P,H = BatchNormalization()(P),BatchNormalization()(H)

    # 3, Three Fully Connected Layer For Recognition
    joint = merge([P,H],mode='concat')
    for i in range(2): #  2 Fully Connected Layer
      joint = Dense(2*self.HiddenSize, activation=self.Activate,
                    W_regularizer=l2(self.L2Strength) if self.L2Strength else None)(joint)
      joint = Dropout(self.DropProb)(joint)
      joint = BatchNormalization()(joint)

    # 5, Final Output of the Model
    pred = Dense(3, activation='softmax')(joint)
    self.model = Model(input=[premise, hypothesis], output=pred)

  @time_count
  def creat_biGRU_model(self):
    assert self.rnn_type == 'biGRU'
    # 1, Sentences Input Layer
    premise = Input(shape=(self.MaxLen,),dtype='int32')
    hypothesis = Input(shape=(self.MaxLen,),dtype='int32')

    # 2, Word Encoder: Fully Connected Layer
    FullyConnected = TimeDistributed(Dense(self.HiddenSize, activation=self.Activate))
    ebdP = FullyConnected(self.Embed(premise))
    ebdH = FullyConnected(self.Embed(hypothesis))

    # 3, Sentence Encoder: Bidirectional GRU
    rnn_kwargs = dict(output_dim=self.HiddenSize, dropout_W=self.DropProb, dropout_U=self.DropProb)
    biGRU = Bidirectional(recurrent.GRU(**rnn_kwargs))
    ebdP, ebdH = biGRU(ebdP), biGRU(ebdH)
    ebdP, ebdH = BatchNormalization()(ebdP),BatchNormalization()(ebdH)

    # 4, Two FC Layer to Recognize Sentence Vector
    joint = merge([ebdP,ebdH],mode='concat')
    for i in range(3): # 2 Layers fully connected layer
      joint = Dense(2*self.HiddenSize,
                    activation=self.Activate,
                    W_regularizer=l2(self.L2Strength))(joint)
      joint = Dropout(self.DropProb)(joint)
      joint = BatchNormalization()(joint)

    # 5, Final Output of the Model
    pred = Dense(3, activation='softmax')(joint)
    self.model =  Model(input=[premise, hypothesis], output=pred)

  @time_count
  def compile_model(self):
    """ Load Possible Existing Weights and Compile the Model """
    fn = self.rnn_type + '.check'
    if os.path.exists(fn):
      self.model.load_weights(fn,by_name=True)
      print('--------Load Weights Successful!--------')
    self.model.compile(optimizer=self.Optimizer,
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    self.model.summary()

  def start_train(self):
    """ Starts to Train the entire Model Based on set Parameters """
    # 1, Prep
    if self.rnn_type == 'sumRNN':  self.creat_sumRNN_model()
    if self.rnn_type == 'biGRU': self.creat_biGRU_model()
    self.compile_model()
    callback = [EarlyStopping(patience=self.Patience),
                ReduceLROnPlateau(patience=5,verbose=1),
                CSVLogger(filename=self.rnn_type+'log.csv'),
                ModelCheckpoint(self.rnn_type + '.check', save_best_only=True, save_weights_only=True)]

    # 2, Train
    self.model.fit(x = [self.train[0],self.train[1]],
                   y = self.train[2],
                   batch_size = self.BatchSize,
                   nb_epoch = self.MaxEpoch,
                   #validation_data = ([self.validation[0],self.validation[1]],self.validation[2]),
                   validation_data=([self.validation[0], self.validation[1]], self.validation[2]),
                   callbacks = callback)

    # 3, Evaluate
    self.model.load_weights(self.rnn_type + '.check') # revert to the best model
    #loss, acc = self.model.evaluate([self.test[0],self.test[1]],self.test[2],batch_size=self.BatchSize)
    loss, acc = self.model.evaluate([self.test[0],self.test[1]],
                                    self.test[2],batch_size=self.BatchSize)

    return loss, acc # loss, accuracy on test data set

  def evaluate_on_test(self):
    loss, acc = self.model.evaluate([self.test[0],self.test[1]],
                                    self.test[2],batch_size=self.BatchSize)
    print("Test: loss = {:.5f}, acc = {:.3f}%".format(loss,acc*100))

  def predict_on_sample(self):
    """ The model must be compiled before execuation """
    prep_alfa = lambda X: pad_sequences(sequences=self.indexer.texts_to_sequences(X),
                                        maxlen=self.MaxLen)
    while True:
      prem = input("Please input the premise:\n")
      hypo = input("Please input another sent:\n")
      unknown = set([word for word in list(filter(lambda x: x != ' ',
                                                  re.split(r'(\W)',prem) + re.split(r'(\W)',hypo)))
                          if word not in self.indexer.word_counts.keys()])
      if unknown:
        print('[WARNING] {} Unregistered Words:{}'.format(len(unknown),unknown))
      prem, hypo = prep_alfa([prem]), prep_alfa([hypo])
      ans = np.reshape(self.model.predict(x=[prem,hypo],batch_size=1),-1)
      print('\n Contradiction \t{:.1f}%\n'.format(float(ans[0])*100),
            'Neutral \t{:.1f}%\n'.format(float(ans[1])*100),
            'Entailment \t{:.1f}%\n'.format(float(ans[2])*100))


md = NLI_USE_RNN(rnn_type='biGRU')
#md.creat_GloVe_weights()
md.prep_data()
md.prep_embd()

#md.creat_sumRNN_model()
md.creat_biGRU_model()
#md.creat_MPM_model()
#md.evaluate_on_test()
#md.start_train()
md.compile_model()
#md.evaluate_on_test()
md.predict_on_sample()


