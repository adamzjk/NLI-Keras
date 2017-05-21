"""





class AttentionRNN(GRU):

  def __init__(self, atten_states, states_len, L2Strength, **kwargs):
    '''
    :param atten_states: previous states for attention
    :param states_len: length of state
    :param L2Strength: for regularization
    :param kwargs: for GRU
    '''
    self.p_states = atten_states
    self.states_len = states_len
    self.size = kwargs['units']
    self.L2Strength = L2Strength
    super(AttentionRNN, self).__init__(**kwargs)

  def build(self, input_shape):
    input_dim = input_shape[-1]
    input_length = input_shape[1]
    self.W1 = self.add_weight(shape = (self.units + input_dim, 1),
                              initializer = 'random_uniform',
                              regularizer=l2(self.L2Strength),
                              trainable = True)
    self.b1 = self.add_weight(shape=(1,),
                              initializer = 'zero',
                              regularizer=l2(self.L2Strength),
                              trainable= True)
    '''
    self.W2 = self.add_weight(shape=(self.units + input_dim, self.units),
                              initializer='random_uniform',
                              regularizer=l2(self.L2Strength),
                              trainable=True)
    self.b2 = self.add_weight(shape=(self.units,),
                              initializer='zero',
                              regularizer=l2(self.L2Strength),
                              trainable=True)
    '''
    super(AttentionRNN, self).build(input_shape)

  def step(self, inputs, states):

    alfa = K.repeat(inputs, self.states_len) # alfa = [batch_size, states_len, units]
    alfa = K.concatenate([self.p_states, alfa], axis = 2) # alfa = [batch_size, states_len, 2*units]
    scores = K.tanh(K.dot(alfa, self.W1) + self.b1) # scores = [batch_size, states_len, 1]
    scores = K.softmax(scores)
    scores = K.reshape(scores, (-1, 1, self.states_len)) # scores = [batch_size, 1, states_len]
    attn = K.batch_dot(scores, self.p_states) # attn = [batch_size, 1, units]
    attn = K.reshape(attn, (-1, self.units))  # attn = [batch_size, units]

    h = keras.layers.add([inputs, attn]) # h = [batch_size, 2*units]
    #h = K.dot(h, self.W2) + self.b2 # h = [batch_size, units]

    h, _ = super(AttentionRNN, self).step(h, [h])  # h = [batch_size, units]
    return h, [h]

  def compute_output_shape(self, input_shape):
    return input_shape[0], self.units


  def create_attention_GRU_model(self):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    set_session(tf.Session(config=config))

    premise = Input(shape=(self.PremMaxLen,), dtype='int32')
    hypothesis = Input(shape=(self.HypoMaxLen,), dtype='int32')

    #FullyConnected = TimeDistributed(Dense(self.HiddenSize, activation=self.Activate))
    #ebdP = FullyConnected(self.Embed(premise))
    #ebdH = FullyConnected(self.Embed(hypothesis))
    ebdP, ebdH = self.Embed(premise), self.Embed(hypothesis)

    biGRU = recurrent.GRU(units=self.HiddenSize,
                          dropout=self.DropProb,
                          return_sequences=True)
    p_states = biGRU(ebdP)

    attenGRU = AttentionRNN(atten_states= p_states,
                            states_len= self.PremMaxLen,
                            L2Strength=self.L2Strength,
                            units = self.HiddenSize,
                            dropout = self.DropProb,
                            return_sequences = False)
    attnout = attenGRU(ebdH, initial_state=p_states[:,-1])

    final = Dense(self.HiddenSize,
                  activation = self.Activate,
                  kernel_regularizer = l2(self.L2Strength),
                  bias_regularizer = l2(self.L2Strength))(attnout)
    final = Dropout(self.DropProb)(final)
    final = BatchNormalization()(final)
    final = Dense(3, activation='softmax')(final)
    self.model = Model(inputs=[premise, hypothesis], outputs=final)


class GetEph(Layer):

  def __init__(self,
               F_p, F_h,
               PremMaxLen,
               HypoMaxLen,
               **kwargs):
    self.F_p = F_p
    self.F_h = F_h
    self.PremMaxLen = PremMaxLen
    self.HypoMaxLen = HypoMaxLen
    super(GetEph, self).__init__(**kwargs)

  def build(self, input_shape):
    super(GetEph, self).build(input_shape)  # Be sure to call this somewhere!

  def call(self, inputs, **kwargs):
    return K.batch_dot(self.F_p, K.reshape(self.F_h, (-1, 512, self.HypoMaxLen)))

  def compute_output_shape(self, input_shape):
    return input_shape[0], self.PremMaxLen, self.HypoMaxLen





def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
  for i, line in enumerate(open(fn)):
    if limit and i > limit:
      break
    data = json.loads(line)
    label = data['gold_label']
    s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
    s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
    if skip_no_majority and label == '-':
      continue
    yield (label, s1, s2)

def get_data(fn, limit=None):
  raw_data = list(yield_examples(fn=fn, limit=limit))
  left = [s1 for _, s1, s2 in raw_data]
  right = [s2 for _, s1, s2 in raw_data]
  print(max(len(x.split()) for x in left))
  print(max(len(x.split()) for x in right))

  LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
  Y = [LABELS[l] for l, s1, s2 in raw_data]
  return left, right, Y

def yield_data(fn):
  # This is the funtion to process original SNLI data set
  for i,line in enumerate(fn):
    temp,new = json.loads(line),{}
    if temp['gold_label'] == '-' : continue
    yield (temp['gold_label'],temp['sentence1'],temp['sentence2'])



#file = open('/Users/Adam/Documents/Data/snli_1.0/snli_1.0_test.jsonl','r')
#open('test.json','w').write(json.dumps(list(yield_data(file))))

'''
training = get_data('snli_1.0_train.jsonl')
validation = get_data('snli_1.0_dev.jsonl')
test = get_data('snli_1.0_test.jsonl')
'''

#Y = np_utils.to_categorical(Y, len(LABELS))
#open('train.json','w').write(json.dumps(training))
#open('validation.json','w').write(json.dumps(validation))
#open('test.json','w').write(json.dumps(test))





"""