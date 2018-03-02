%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.learner import *

import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling

from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *

import dill as pickle

# ---------------------------------------- #

# download the movie dataset
# this is the sentiment classification task consists of predicting
# the polarity of the given text(positive or negative)

PATH='data/aclImdb'
TRN_PATH = 'train/all'
VAL_PATH = 'test/all'

TRN = f'{PATH}{TRN_PATH}'
VAL = f'{PATH}{VAL_PATH}'

%ls {PATH}

# ----------------------------------------- #

