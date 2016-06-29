#!/usr/bin/env python

import pickle, argparse, time
import numpy as np
from datetime import datetime
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

args = argparse.ArgumentParser(
        description="Generate text with a HMM")
args.add_argument("-l", "--num-lines", type=int, default=1,
        help="number of lines to generate")
args.add_argument("-w", "--num-words", type=int, default=25,
        help="number of words per line")
args.add_argument("--seed", type=int, default=datetime.now().microsecond,
        help="seed number to configure repeatable random generation")
args.add_argument("filename", 
        help="input filename without extension (assumes FILENAME.pkl and FILENAME.le)")
args = args.parse_args()

with open("{0}.le".format(args.filename), "rb") as f:
    le = pickle.load(f)

model = joblib.load("{0}.pkl".format(args.filename))

seed = args.seed

for _i in range(args.num_lines):
    symbols, _states = model.sample(args.num_words, random_state=seed)
    seed = seed + 1

    output = le.inverse_transform(np.squeeze(symbols))
    for word in output:
        print(word, end=" ")
    print("\n")
