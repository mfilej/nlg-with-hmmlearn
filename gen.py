#!/usr/bin/env python

import pickle, argparse, time, random, sys
import numpy as np
from datetime import datetime
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

args = argparse.ArgumentParser(
        description="Generate text with a HMM")
args.add_argument("-l", "--num-lines", type=int, default=1,
        help="number of lines to generate")

group = args.add_mutually_exclusive_group(required=True)
group.add_argument("-w", "--num-words", type=int, default=25,
        help="number of words per line (excludes -d)")
group.add_argument("-d", "--num-words-distribution", type=argparse.FileType("r"),
        help="file containing numbers of words per line (one number per line; alternative to -w)")

args.add_argument("--seed", type=int, default=datetime.now().microsecond,
        help="seed number to configure repeatable random generation")
args.add_argument("filename", 
        help="input filename without extension (assumes FILENAME.pkl and FILENAME.le)")
args = args.parse_args()

if args.num_words_distribution:
    len_dist = [int(line) for line in args.num_words_distribution]
    def num_words(seed):
        random.seed(seed)
        return random.choice(len_dist)
else:
    def num_words(seed):
        return args.num_words


with open("{0}.le".format(args.filename), "rb") as f:
    le = pickle.load(f)
model = joblib.load("{0}.pkl".format(args.filename))

seed = args.seed
for _i in range(args.num_lines):
    random_len = num_words(seed=seed)
    seed = seed + 1

    symbols, _states = model.sample(random_len, random_state=seed)

    output = le.inverse_transform(np.squeeze(symbols))
    for word in output:
        print(word, end=" ")
    print()

print("seed={0}".format(args.seed), file=sys.stderr)
