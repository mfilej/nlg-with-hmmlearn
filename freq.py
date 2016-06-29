#!/usr/bin/env python

import pickle, argparse
import numpy as np
from nltk import FreqDist
from sklearn.preprocessing import LabelEncoder

args = argparse.ArgumentParser(
        description="Generate text using relative frequencies for words in training set")
args.add_argument("-l", "--num-lines", type=int, default=1,
        help="number of lines to generate")
args.add_argument("-w", "--num-words", type=int, default=25,
        help="number of words per line")
# args.add_argument("--seed", type=int, default=datetime.now().microsecond,
#         help="seed number to configure repeatable random generation")
args.add_argument("freqdist", 
        help="path to .freqdist file")
args.add_argument("le", 
        help="path to .le file")
args = args.parse_args()

with open(args.freqdist, "rb") as f:
    fd = pickle.load(f)
with open(args.le, "rb") as f:
    le = pickle.load(f)

key_prob_pairs = ((key, fd.freq(key)) for key in fd.keys())

keys, probs = zip(*key_prob_pairs)

for _line in range(args.num_lines):
    sample = np.random.choice(keys, size=args.num_words, p=probs)
    output = le.inverse_transform(sample)
    print(*output, end="\n\n")
