#!/usr/bin/env python

import pickle, argparse, random, sys
from datetime import datetime
import numpy as np
from nltk import FreqDist
from sklearn.preprocessing import LabelEncoder

args = argparse.ArgumentParser(
        description="Generate text using relative frequencies for words in training set")
args.add_argument("-l", "--num-lines", type=int, default=1,
        help="number of lines to generate")

group = args.add_mutually_exclusive_group(required=True)
group.add_argument("-w", "--num-words", type=int, default=25,
        help="number of words per line (excludes -d)")
group.add_argument("-d", "--num-words-distribution", type=argparse.FileType("r"),
        help="file containing numbers of words per line (one number per line; alternative to -w)")

args.add_argument("--seed", type=int, default=datetime.now().microsecond,
        help="seed number to configure repeatable random generation")
args.add_argument("freqdist", 
        help="path to .freqdist file")
args.add_argument("le", 
        help="path to .le file")
args = args.parse_args()

if args.num_words_distribution:
    len_dist = [int(line) for line in args.num_words_distribution]
    def num_words(seed):
        random.seed(seed)
        return random.choice(len_dist)
else:
    def num_words(seed):
        return args.num_words

with open(args.freqdist, "rb") as f:
    fd = pickle.load(f)
with open(args.le, "rb") as f:
    le = pickle.load(f)

key_prob_pairs = ((key, fd.freq(key)) for key in fd.keys())

keys, probs = zip(*key_prob_pairs)

seed = args.seed
for _line in range(args.num_lines):
    random_len = num_words(seed=seed)
    seed = seed + 1

    sample = np.random.choice(keys, size=random_len, p=probs)
    output = le.inverse_transform(sample)
    print(*output, end="\n")

print("seed={0}".format(args.seed), file=sys.stderr)
