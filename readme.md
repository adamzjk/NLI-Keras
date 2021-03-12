# SNLI-RTE in Keras

​	This repository contains 2 models. 1, Decomposable Attention Model by Ankur P. Parikh et al. 2016; and 2, Enhanced LSTM Attention model by Qian Chen et al. 2016; The first model achieves **86%** accuracy with only **500k** parameters whereas the second model achieves **88%** accuracy with **4.6m** parameters.

## Requirements

- Python3
- Tensorflow v1.0
- Keras v2

## Dataset

​	You can use either Stanford Natural Language Inference(**SNLI**) dataset or Recognizing Textual Entailment(**RTE**) dataset, some preprocessing methods are **not** shown in this repository. Just specifiy data set and model name, model will adjust itself:

```python
if __name__ == '__main__': # easy!
  md = AttentionAlignmentModel(annotation='EAM', dataset='snli')
  md.prep_data()
  md.prep_embd()
  md.create_enhanced_attention_model() # or  # md.create_standard_attention_model()
  md.compile_model()
  md.start_train()
  md.evaluate_on_test()
```

then 

```bash
python3 tfRNN.py
```

## Attention Visualization

![alt text](http://wx1.sinaimg.cn/large/98d135cfly1fft8uc9eucj20rs0jggn8.jpg)

​	This is done by letting attention weight as an output when testing our model. Heatmap is drawn with the help of ``seaborn``.

## Interactive Prediction

​	Function ``interactive_predict`` let user input 2 sentences and return 3 probabilities w.r.t. entailment, contradiction and neutral. If ``test_mode`` is set, then the attention visualization heatmap will also be saved as file.



