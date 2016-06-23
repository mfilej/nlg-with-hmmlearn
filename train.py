import sys, json, codecs, pickle
import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from hmmlearn import hmm

np.random.seed(seed=None)

num_states = int(sys.argv[1])
input_path = sys.argv[2]

lines = [line.split() for line in codecs.open(input_path, "r", encoding="utf8")]
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
