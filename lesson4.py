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

# lets look inside the training folder

trn_files = !ls {TRN}
trn_files[:10]

# and at an example review
review = !cat {TRN}{trn_files[4]}
review[0]

# ----------------------------------------- #

# now i'll check howmany words are in the dataset

!find {TRN} -name '*.txt' | xargs cat | wc -w

!find {VAL} -name '*.txt' | xargs cat | wc -w

# Before we analyze text, we must first tokenize it. THis refers to the process
# of splitting a sentense into an array of words ( more generally into tokens )

" ".join(spacy_tok(review[0]))

# here spacy.io's tokenizer is used

TEXT = data.Field(lower=True, tokenize=spacy_tok)

# fastai works closely with torchtext. We create a ModelData object for language 
# modeling by taking advantage of LanguageModelData, passing it our torchtext 
# field object, and the paths to our training, test, validation sets.
# Here as we do not have a separate validation set we'll use test set for both test and validation

# batch size
bs = 64; 
# back propagation through time
# this defines how many layers we'll backprop through.
# Making this number higher will increase time and memory requirements,
# But larger number will also improve models ability to handle long sentences.
bptt = 70;

FILES = dict(train=TRN_PATH, validation=VAL_PATH, test=VAL_PATH)
md = LanguageModelData.from_text_files(
    PATH,
    TEXT,
    **FILES,
    bs=bs,
    bptt=bptt,
    min_freq=10
)


# --------- #

# After building our `ModelData` object, it automatically fills the TEXT obhect
# with a very important attribute TEXT.vocab.
# This is a vocabulary, which stores which stores which words have been seen
# in the text, and how each will be mapped to unique integer id. We'll
# need to use this information again later, so we save it.

pickle.dump(TEXT, open(f'{PATH}models/TEXT.pkl','wb'))

# here are the # batches; # unique tokens in the vocab;
len(md.trn_dl), md.nt, len(md.trn_ds), len(md.trn_ds[0].text)


# this is the start of the mapping from integer IDs to unique tokens
# 'itos' : 'int-to-string'
TEXT.vocab.itos[:12]

# 'stoi'
TEXT.vocab.stoi['the']


# Note that in a LanguageModelData object there is only one item in each dataset:
# all the words of the text joined together.
md.trn_ds[0].text[:12]


# torch text will handle turning this words into integer ids for us automatically

TEXT.numeriacalize([md.trn_ds[0].text[:12]])

# Our LanguageModelData object will create batches with 64 columns (that's our batch size)
# and varying sequence lengths of around 80 tokens (that's our bptt parameter -
# backprop through time)

# Each batch also contains the exact same data as labels, but one word later in the text -
# since we're trying to always predict the next word. The labels are flattened into 1d array

next(iter(md.trn_dl))


### ---------- ###

# Train

# we have a number of prameters to set - we'll learn more about these later
# but these values are suitable for many problems.

em_sz = 200     # size of each embedding vector
nh = 500        # number of hidden activations per layer
nl = 3          # number of layers

# Need to create our version of Adam Optimizer with less momentum than it's default of 0.9

opt_fn = partial(optim.Adam, betas=(0.7, 0.99))

# please read comment after cell 18 in original notebook

# However, the other prameters (alpha, beta and clip) shouldn't generally need tuning.
learner = md.get_model(
    opt_fn,
    em_sz,
    nh,
    nl,
    dropouti=0.05,
    dropout=0.05,
    wdrop=0.1,
    dropoute=0.02,
    dropouth=0.05)

learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learner.clip=0.3

learner.fit(3e-3, 4, wds=1e-6, cycle_len=1, cycle_mult=2)

learner.save_encoder('adam1_enc')

learner.load_encoder('adam1_enc')


learner.load_cycle('adam3_10', 2)
learner.fit(3e-3, 4, wds=1e-6, cycle_len=10)

learner.save_encoder('adam3_10_enc')
learner.save_encoder('adam3_20_enc')
learner.load_encoder('adam3_20_enc')
pickle.dump(TEXT, open(f'{PATH}models/TEXT.pkl','wb'))


# TEST

# We can play around with our language model a bit to check it seems to be
# working OK. First, create a short bit of text to prime a set of predictions
# We'll use our torchtext field to numericalize it so we can feed it to our
# language model.

m = learner.model
ss=""". So, it wasn't quite was I was expecting, but I really liked it anyway! The best"""

s = (spacy_tok(ss))

t = TEXT.numericalize(s)
' '.join(s[0])


# -------- #

# we haven't yer added methods to make it easy to test a language model
# following is manual process to do that.

# set batch size to 1
m[0].bs = 1

# turn off dropout
m.eval()

#reset hidden state
m.reset()

# get predictions from model
res,*_ = m(t)

# put the batch size to what it was
m[0].bs = bs

# lets see what the top 10 prediction were for the next word after our short text
nexts = torch.topk(res[-1], 10)[1]
(TEXT.vocab.itos[o] for o in to_np(nexts))

# ------------ #
# let's see if the model can generate a bit more text

print(ss,"\n")
for i in range(50):
    n=res[-1].topk(2)[1]
    n = n[1] if n.data[0]==0 else n[0]
    print(TEXT.vocab.itos[n.data[0]], end=' ')
    res,*_ = m(n[0].unsqueeze(0))
print('...')

# ------------ #
