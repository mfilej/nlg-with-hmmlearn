#!/usr/bin/env python

import pickle, argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder

args = argparse.ArgumentParser(
        description="Generate text with a HMM")
args.add_argument("-l", "--num-lines", type=int, default=1,
        help="number of lines to generate")
args.add_argument("-w", "--num-words", type=int, default=25,
        help="number of words per line")
# args.add_argument("--seed", type=int, default=datetime.now().microsecond,
#         help="seed number to configure repeatable random generation")
args.add_argument("input", 
        help="path to INPUT.le file")
args = args.parse_args()

with open(args.input, "rb") as f:
    le = pickle.load(f)

words = le.classes_

for _line in range(args.num_lines):
    sample = np.random.choice(words, args.num_words)
    print(*sample, end="\n\n")
