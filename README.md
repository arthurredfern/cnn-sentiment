[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# Sentiment classification with CNNs

:warning: Work in progress :warning:


This repository contains an implementation based on the paper *Convolutional Neural Networks for Sentence Classification* by Yoon Kim.

## Data

The model is trained with the Customer Reviews dataset, annotated by Minqing Hu and Bing Liu at the University of Illinois. More information can be found [here](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html). Additionally, the model uses pretrained word embeddings obtained with [GloVe](https://nlp.stanford.edu/projects/glove/).

To download the data and embeddings, run

```shell
bash get-data.sh
```
