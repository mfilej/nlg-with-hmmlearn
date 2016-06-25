#!/usr/bin/env python
import sys, json, codecs, pickle, argparse
import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from hmmlearn import hmm

np.random.seed(seed=None)

args = argparse.ArgumentParser(
        description="Train discrete HMM based on given input text")
args.add_argument("-n", "--num-states", type=int, required=True,
        help="number of hidden states")
args.add_argument("--init", choices=["builtin", "freq", "flat"], default="builtin",
        help="strategy for estimating initial model parameters")
args.add_argument("input", nargs="?", type=argparse.FileType("r"),
        default=sys.stdin,
        help="input text that will act as training set for the model")
args = args.parse_args()

lines = [line.split() for line in args.input]
words = [word.lower() for line in lines for word in line]

alphabet = set(words)
le = LabelEncoder()
le.fit(list(alphabet))

seq = le.transform(words)
seq = np.fromiter(seq, np.int64)
seq = np.atleast_2d(seq).T

model = hmm.MultinomialHMM(n_components=num_states)

lengths = [len(line) for line in lines]
model = model.fit(seq, lengths)

joblib.dump(model, "hmm.pkl")
with open("hmm.le", "wb") as f:
    pickle.dump(le, f)
