#!/usr/bin/env python
import sys, json, codecs, pickle, argparse
import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from hmmlearn import hmm
from nltk import FreqDist

def warn(msg):
    print(msg, file=sys.stderr)

np.random.seed(seed=None)

args = argparse.ArgumentParser(
        description="Train discrete HMM based on given input text")
args.add_argument("-n", "--num-states", type=int, required=True,
        help="number of hidden states")
args.add_argument("--init", choices=["builtin", "freq", "flat"], default="builtin",
        help="strategy for estimating initial model parameters")
args.add_argument("-o", required=True, metavar="FILENAME",
        help="file basename for output files")
args.add_argument("input", nargs="?", type=argparse.FileType("r"),
        default=sys.stdin,
        help="input text that will act as training set for the model (can be passed on stdin)")
args = args.parse_args()

lines = [line.split() for line in args.input]
words = [word.lower() for line in lines for word in line]

alphabet = set(words)
le = LabelEncoder()
le.fit(list(alphabet))

seq = le.transform(words)
features = np.fromiter(seq, np.int64)
features = np.atleast_2d(features).T
fd = FreqDist(seq)

def outfile(ext):
    return "{name}.{init}.{n}.{ext}".format(
            name=args.o, init=args.init, n=args.num_states, ext=ext)

def builtin():
    warn("Initial parameter estimation using built-in method")
    model = hmm.MultinomialHMM(n_components=args.num_states, init_params='ste')
    return model

def frequencies():
    warn("Initial parameter estimation using relative frequencies")

    frequencies = np.fromiter((fd.freq(i) for i in range(len(alphabet))), dtype=np.float64)
    emission_prob = np.stack([frequencies] * args.num_states)
    print(repr(emission_prob.shape))

    model = hmm.MultinomialHMM(n_components=args.num_states, init_params='st')
    model.emissionprob_ = emission_prob
    return model

def flat():
    return None

def dispatch_init_est(fun):
    return {
        "builtin": builtin,
        "freq": frequencies,
        "flat": flat,
    }.get(fun, builtin)

model = dispatch_init_est(args.init)()

lengths = [len(line) for line in lines]
model = model.fit(features, lengths)

joblib.dump(model, outfile("pkl"))
with open(outfile("le"), "wb") as f:
    pickle.dump(le, f)
with open(outfile("freqdist"), "wb") as f:
    pickle.dump(fd, f)

warn("Output written to:\n\t- {0}\n\t- {1}\n\t- {2}".format(
        outfile("pkl"), outfile("le"), outfile("freqdist")
    ))
