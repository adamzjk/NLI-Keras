# SNLI in Keras

​	This repository contains 2 models. 1, Decomposable Attention Model by Ankur P. Parikh et al. 2016; and 2, Enhanced LSTM Attention model by Qian Chen et al. 2016; The first model achieves **86%** accuracy with only **500k** parameters whereas the second model achieves **88%** accuracy with **4.6m** parameters.

## Requirements

- Python3
- Tensorflow v1.0
- Keras v2

## Data

​	I did some preprocessing on the original data and use it to train the neural network, which means you can't run this model directly with the original SNLI database. It is recommended to read the essential code which lies in ``create_xxx_model()`` method in file ``tfRNN.py`` and build your own network.

## Attention Visualization

![alt text](http://wx1.sinaimg.cn/large/98d135cfly1fft8uc9eucj20rs0jggn8.jpg)

​	This is done by letting attention weight as an output when testing our model. Heatmap is drawn with the help of ``seaborn``.

## Interactive Prediction

​	Function ``interactive_predict`` let user input 2 sentences and return 3 probabilities w.r.t. entailment, contradiction and neutral. If ``test_mode`` is set, then the attention visualization heatmap will also be saved as file.

## Resources

​	If you need preprocessed data, compressed GloVe word embeddings, well-trained weights, please contact adamzjk@foxmail.com



