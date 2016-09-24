#!/usr/bin/env python

import pickle, argparse, random, sys
from datetime import datetime

import numpy as np
from sklearn.preprocessing import LabelEncoder

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
args.add_argument("input", 
        help="path to INPUT.le file")
args = args.parse_args()

if args.num_words_distribution:
    len_dist = [int(line) for line in args.num_words_distribution]
    def num_words(seed):
        random.seed(seed)
        return random.choice(len_dist)
else:
    def num_words(seed):
        return args.num_words

with open(args.input, "rb") as f:
    le = pickle.load(f)

words = le.classes_

seed = args.seed
for _line in range(args.num_lines):
    target_len = num_words(seed=seed)
    seed = seed + 1

    sample = np.random.choice(words, target_len)
    print(*sample, end="\n")

print("seed={0}".format(args.seed), file=sys.stderr)
